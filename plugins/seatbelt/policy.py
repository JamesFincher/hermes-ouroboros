"""Seatbelt policy engine.

Declarative rules evaluated on every ``pre_tool_call`` hook.  Rules can
veto a call (``block``), escalate it to the human approval gate
(``approve``), or simply flag it in the audit trail (``audit``).  Two
stateful rule kinds are supported:

  - rate:   fires after `count` matches inside `per_seconds` (per process)
  - budget: fires after `max_calls` matches inside the current session

The engine returns directives in the exact shape the Hermes plugin host
expects from ``pre_tool_call`` callbacks::

    {"action": "block",   "message": "..."}
    {"action": "approve", "message": "...", "rule_key": "..."}

Everything is fail-open on malformed rules (a broken rule logs and is
skipped) and fail-closed nowhere: seatbelt never blocks unless a rule
explicitly says so.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    name: str
    tools: List[str]
    action: str = "audit"
    message: str = ""
    rule_key: str = ""
    args_regex: Optional[str] = None
    rate_count: int = 0
    rate_per_seconds: int = 0
    budget_max_calls: int = 0
    enabled: bool = True
    # runtime state (not serialized)
    _compiled: Any = field(default=None, repr=False)
    _window_hits: List[float] = field(default_factory=list, repr=False)
    _session_hits: Dict[str, int] = field(default_factory=dict, repr=False)

    def matches_tool(self, tool_name: str) -> bool:
        return any(fnmatch.fnmatchcase(tool_name, pat) for pat in self.tools)

    def matches_args(self, args_json: str) -> bool:
        if not self.args_regex:
            return True
        if self._compiled is None:
            try:
                self._compiled = re.compile(self.args_regex)
            except re.error as exc:
                logger.warning("seatbelt: rule %s has bad regex: %s", self.name, exc)
                self._compiled = False
        if not self._compiled:
            return False
        return bool(self._compiled.search(args_json))

    def rate_exceeded(self, now: float) -> bool:
        if self.rate_count <= 0 or self.rate_per_seconds <= 0:
            return False
        cutoff = now - self.rate_per_seconds
        self._window_hits = [t for t in self._window_hits if t >= cutoff]
        return len(self._window_hits) >= self.rate_count

    def budget_exceeded(self, session_id: str) -> bool:
        if self.budget_max_calls <= 0:
            return False
        return self._session_hits.get(session_id, 0) >= self.budget_max_calls

    def record_hit(self, session_id: str, now: float) -> None:
        if self.rate_count > 0:
            self._window_hits.append(now)
        self._session_hits[session_id] = self._session_hits.get(session_id, 0) + 1


class PolicyEngine:
    """Loads rules.yaml and evaluates tool calls against it."""

    def __init__(self, rules_path: Path):
        self.rules_path = Path(rules_path)
        self.rules: List[Rule] = []
        self.paused = False
        self.loaded_at: float = 0.0
        self.load()

    # ------------------------------------------------------------------ io

    def load(self) -> None:
        import yaml  # hermes bundles pyyaml (gateway hooks use it)

        if not self.rules_path.exists():
            self.rules = []
            return
        try:
            data = yaml.safe_load(self.rules_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("seatbelt: failed to parse %s: %s", self.rules_path, exc)
            return
        rules: List[Rule] = []
        for raw in data.get("rules") or []:
            try:
                rules.append(self._parse_rule(raw))
            except Exception as exc:
                logger.warning("seatbelt: skipping malformed rule %r: %s", raw, exc)
        self.rules = rules
        self.loaded_at = time.time()

    @staticmethod
    def _parse_rule(raw: Dict[str, Any]) -> Rule:
        rate = raw.get("rate") or {}
        budget = raw.get("budget") or {}
        return Rule(
            name=str(raw.get("name") or f"rule-{id(raw)}"),
            tools=list(raw.get("tools") or ["*"]),
            action=str(raw.get("action") or "audit"),
            message=str(raw.get("message") or ""),
            rule_key=str(raw.get("rule_key") or ""),
            args_regex=raw.get("args_regex"),
            rate_count=int(rate.get("count") or 0),
            rate_per_seconds=int(rate.get("per_seconds") or 0),
            budget_max_calls=int(budget.get("max_calls") or 0),
            enabled=bool(raw.get("enabled", True)),
        )

    # ---------------------------------------------------------- evaluation

    def evaluate(self, tool_name: str, args: Dict[str, Any], session_id: str) -> Optional[Dict[str, Any]]:
        """Return a pre_tool_call directive dict, or None to allow silently.

        Evaluation order: block rules first (hard vetoes win), then approve
        rules (rate/budget/stateful and simple), so a block can never be
        masked by an earlier approval request.  Audit-only rules never
        produce a directive.
        """
        if self.paused:
            return None
        session_id = session_id or "_nosession"
        try:
            args_json = json.dumps(args or {}, default=str, ensure_ascii=False)
        except Exception:
            args_json = str(args)

        now = time.time()
        approving: Optional[Dict[str, Any]] = None

        for rule in self.rules:
            if not rule.enabled or not rule.matches_tool(tool_name):
                continue
            if not rule.matches_args(args_json):
                continue

            fired = True
            if rule.rate_count > 0 or rule.budget_max_calls > 0:
                # Stateful rule: only fires once the threshold is crossed.
                fired = rule.rate_exceeded(now) or rule.budget_exceeded(session_id)
            if fired and rule.action == "block":
                rule.record_hit(session_id, now)
                return {
                    "action": "block",
                    "message": rule.message or f"Seatbelt blocked {tool_name} (rule: {rule.name}).",
                    "_rule": rule.name,
                }
            if fired and rule.action == "approve" and approving is None:
                approving = {
                    "action": "approve",
                    "message": rule.message or f"Seatbelt requires approval for {tool_name} (rule: {rule.name}).",
                    "rule_key": rule.rule_key or f"seatbelt:{rule.name}",
                    "_rule": rule.name,
                }
            # Every matching rule accrues usage, even when it doesn't fire
            # yet — that's what makes the threshold meaningful next time.
            rule.record_hit(session_id, now)

        return approving

    # -------------------------------------------------------------- introspection

    def status(self) -> Dict[str, Any]:
        return {
            "rules_path": str(self.rules_path),
            "rules_loaded": len(self.rules),
            "rules_enabled": sum(1 for r in self.rules if r.enabled),
            "paused": self.paused,
            "loaded_at": self.loaded_at,
            "stateful_rules": [
                {
                    "name": r.name,
                    "rate": f"{r.rate_count}/{r.rate_per_seconds}s" if r.rate_count else "",
                    "budget": r.budget_max_calls or "",
                    "window_hits": len(r._window_hits),
                    "session_hits": dict(r._session_hits),
                }
                for r in self.rules
                if r.rate_count > 0 or r.budget_max_calls > 0
            ],
        }
