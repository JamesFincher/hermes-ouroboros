"""Seatbelt — adaptive tool-policy engine for Hermes.

Hooks into the ``pre_tool_call`` directive channel (block / approve with
rule_key) that Hermes reserves for policy plugins, and records every
tool call into a local SQLite audit trail via ``post_tool_call``.

Why this is different from the bundled ``security-guidance`` plugin:
security-guidance pattern-matches *file contents* on three write tools
and appends a warning.  Seatbelt is a full policy layer over *every*
tool: declarative YAML rules, per-tool rate limits, per-session call
budgets (runaway-loop guard), human-approval escalation for any tool,
and a queryable 90-day audit trail.

Commands:
    /seatbelt status      — engine state + audit counts
    /seatbelt rules       — list loaded rules
    /seatbelt tail [n]    — last n audited calls
    /seatbelt top [days]  — most-used tools with flagged counts
    /seatbelt reload      — re-read rules.yaml
    /seatbelt pause       — suspend enforcement (audit keeps running)
    /seatbelt resume      — resume enforcement
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_engine = None
_audit = None
_pending_decisions: dict = {}  # (tool_call_id or "tool:ts") -> (decision, rule)


def _paths():
    from hermes_constants import get_hermes_home

    base = Path(get_hermes_home()) / "seatbelt"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _ensure():
    """Lazily create the engine + audit log and seed default rules."""
    global _engine, _audit
    if _engine is not None:
        return
    from .audit import AuditLog
    from .policy import PolicyEngine

    base = _paths()
    rules_path = base / "rules.yaml"
    if not rules_path.exists():
        shutil.copyfile(_HERE / "rules.default.yaml", rules_path)
    _engine = PolicyEngine(rules_path)
    _audit = AuditLog(base / "audit.db")


# ---------------------------------------------------------------- hooks

def _pre_tool_call(tool_name: str = "", args: dict | None = None,
                   session_id: str = "", tool_call_id: str = "", **kwargs):
    _ensure()
    directive = _engine.evaluate(tool_name, args or {}, session_id)
    if directive:
        rule = directive.pop("_rule", "")
        _pending_decisions[tool_call_id or f"{tool_name}:{time.time()}"] = (
            directive["action"], rule)
        _audit.record(tool_name=tool_name, args=args, session_id=session_id,
                      decision=directive["action"], rule=rule)
        return directive
    _pending_decisions[tool_call_id or f"{tool_name}:{time.time()}"] = ("allow", "")
    return None  # observer: allow silently


def _post_tool_call(tool_name: str = "", args: dict | None = None,
                    session_id: str = "", tool_call_id: str = "",
                    duration_ms: float = 0.0, status: str = "", **kwargs):
    _ensure()
    key = tool_call_id or ""
    decision, rule = _pending_decisions.pop(key, None) or (None, None)
    if decision is None:
        # No pre-hook entry (older hermes, or directive consumed): allow.
        decision, rule = "allow", ""
    if decision == "allow":
        _audit.record(tool_name=tool_name, args=args, session_id=session_id,
                      status=status, decision=decision, rule=rule,
                      duration_ms=duration_ms)


# --------------------------------------------------------------- command

def _fmt_ts(ts: float) -> str:
    return time.strftime("%m-%d %H:%M:%S", time.localtime(ts))


def _cmd_seatbelt(raw_args: str) -> str:
    _ensure()
    parts = (raw_args or "").strip().split()
    sub = parts[0].lower() if parts else "status"

    if sub == "status":
        st = _engine.status()
        counts = _audit.counts()
        return (
            f"Seatbelt {'PAUSED (audit only)' if st['paused'] else 'enforcing'}\n"
            f"Rules: {st['rules_enabled']}/{st['rules_loaded']} enabled "
            f"({st['rules_path']})\n"
            f"Audit: {counts['total_events']} events "
            f"({counts['flagged_events']} flagged) — {counts['db_path']}\n"
            + ("\nStateful rules:\n" + "\n".join(
                f"  {r['name']}: rate {r['rate']} budget {r['budget']} "
                f"hits now {r['window_hits']}" for r in st["stateful_rules"])
               if st["stateful_rules"] else "")
        )

    if sub == "rules":
        lines = []
        for r in _engine.rules:
            flags = []
            if r.rate_count:
                flags.append(f"rate {r.rate_count}/{r.rate_per_seconds}s")
            if r.budget_max_calls:
                flags.append(f"budget {r.budget_max_calls}/session")
            if r.args_regex:
                flags.append(f"regex /{r.args_regex[:40]}/")
            lines.append(
                f"{'ON ' if r.enabled else 'off'} [{r.action:7}] {r.name} "
                f"-> {','.join(r.tools)} {' '.join(flags)}")
        return "Seatbelt rules:\n" + ("\n".join(lines) or "  (none loaded)")

    if sub == "tail":
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 15
        rows = _audit.tail(n)
        if not rows:
            return "Seatbelt audit is empty."
        return "Recent tool calls (newest first):\n" + "\n".join(
            f"{_fmt_ts(r['ts'])}  {r['tool_name']:<22} {r['decision']:<8} "
            f"{r['status'] or '-':<8} {r['rule'] or ''}".rstrip()
            for r in rows)

    if sub == "top":
        days = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 7
        rows = _audit.top_tools(days)
        if not rows:
            return f"No tool calls audited in the last {days} days."
        return (f"Tool usage, last {days} days:\n"
                + "\n".join(f"{name:<24} {cnt:>5} calls  avg {avg}ms  flagged {flag}"
                            for name, cnt, avg, flag in rows))

    if sub == "reload":
        _engine.load()
        return f"Seatbelt reloaded {len(_engine.rules)} rules from {_engine.rules_path}"

    if sub == "pause":
        _engine.paused = True
        return "Seatbelt enforcement paused — calls are audited but never blocked/approved. /seatbelt resume to re-enable."

    if sub == "resume":
        _engine.paused = False
        return "Seatbelt enforcement resumed."

    if sub == "prune":
        removed = _audit.prune()
        return f"Pruned {removed} audit events older than 90 days."

    return ("Unknown subcommand. Usage: /seatbelt "
            "[status|rules|tail [n]|top [days]|reload|pause|resume|prune]")


# -------------------------------------------------------------- register

def register(ctx):
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_command(
        "seatbelt",
        _cmd_seatbelt,
        description="Tool-policy engine: status, rules, audit tail, pause/resume",
        args_hint="[status|rules|tail|top|reload|pause|resume|prune]",
    )
