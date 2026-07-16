"""Autopilot miner — detect recurring user intents, propose automations.

Pipeline (read-only against state.db until a suggestion is approved):

  1. Harvest recent *interactive* user messages (cron/subagent/tool
     sessions excluded — automation talking to itself is noise).
  2. One structured LLM pass (ctx.llm.complete_structured, host-owned so
     no extra keys) clusters them into recurring intents and drafts an
     automation for each: a cron schedule + self-contained prompt.
  3. Suggestions are deduped by fingerprint against (a) already-pending
     suggestions, (b) dismissed ones, (c) existing cron job names.
  4. Approving writes a REAL cron job through cron.jobs.create_job — the
     exact function the built-in cronjob tool uses, so the scheduler,
     `hermes cron list` and delivery all work with zero new machinery.

Storage: ~/.hermes/autopilot/{suggestions.json,pending.json,state.json}
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MINE_AFTER_SESSIONS = 5       # sessions to accumulate before auto-mining
_COOLDOWN_S = 6 * 3600         # never auto-mine more often than this

_SUGGESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "examples": {"type": "array", "items": {"type": "string"}},
                    "approx_count": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["cron", "skill", "memory"]},
                    "schedule": {"type": "string"},
                    "draft_prompt": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["title", "rationale", "approx_count", "kind",
                             "draft_prompt", "confidence"],
            },
        }
    },
    "required": ["patterns"],
}

_MINE_INSTRUCTIONS = """You analyze a numbered list of real user messages sent to a personal AI agent over the past days (across CLI and chat platforms). Find RECURRING intents — things the user asked for at least 3 times, or on a detectable rhythm (daily standup summary, weekly review, morning briefing, repeated lookup, repeated rewrite/transformation).

For each pattern:
- title: short imperative name (e.g. "Morning GitHub triage briefing")
- rationale: one sentence of evidence
- examples: up to 3 verbatim (truncated) user messages
- approx_count: how many messages match
- kind: "cron" for anything schedulable (the usual case), "skill" if the user keeps explaining the same procedure (worth a reusable skill), "memory" if it's a standing preference/fact worth pinning
- schedule: for cron — a Hermes schedule string ("every day 08:30", "every monday 09:00", "every 2h"); omit otherwise
- draft_prompt: for cron — a FULLY self-contained prompt the agent can run unattended (no "as we discussed"); for skill — an outline; for memory — the fact to pin
- confidence: 0-1

Only report patterns with approx_count >= 3. If nothing qualifies, return {"patterns": []}. Never invent patterns from single messages."""


def _fingerprint(text: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return hashlib.sha1(norm.encode()).hexdigest()[:12]


class SuggestionStore:
    def __init__(self, base: Path):
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base / "suggestions.json"
        self.state_path = self.base / "state.json"
        self.pending_path = self.base / "pending.json"

    def _read(self, path: Path, default):
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return default

    def _write(self, path: Path, data) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    # suggestions
    def suggestions(self) -> List[Dict[str, Any]]:
        return self._read(self.path, [])

    def save_suggestions(self, items: List[Dict[str, Any]]) -> None:
        self._write(self.path, items)

    def get(self, sug_id: str) -> Optional[Dict[str, Any]]:
        for s in self.suggestions():
            if s.get("id") == sug_id:
                return s
        return None

    def update(self, sug_id: str, **fields) -> bool:
        items = self.suggestions()
        for s in items:
            if s.get("id") == sug_id:
                s.update(fields)
                self.save_suggestions(items)
                return True
        return False

    # auto-mine state
    def state(self) -> Dict[str, Any]:
        return self._read(self.state_path, {})

    def save_state(self, st: Dict[str, Any]) -> None:
        self._write(self.state_path, st)

    # pending session queue
    def pending(self) -> List[str]:
        return self._read(self.pending_path, [])

    def save_pending(self, ids: List[str]) -> None:
        self._write(self.pending_path, ids[-50:])


def _existing_cron_names() -> set:
    try:
        from hermes_constants import get_hermes_home

        jobs_file = Path(get_hermes_home()) / "cron" / "jobs.json"
        data = json.loads(jobs_file.read_text(encoding="utf-8"))
        jobs = data.get("jobs", data) if isinstance(data, dict) else data
        return {str(j.get("name", "")).lower() for j in jobs if isinstance(j, dict)}
    except Exception:
        return set()


def harvest_user_messages(days: int = 14, limit: int = 140) -> List[Dict[str, Any]]:
    from hermes_constants import get_hermes_home

    db_path = Path(get_hermes_home()) / "state.db"
    if not db_path.exists():
        return []
    since = time.time() - days * 86400
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    try:
        cur = conn.execute(
            """
            SELECT m.content, s.source, m.timestamp
            FROM messages m JOIN sessions s ON s.id = m.session_id
            WHERE m.role = 'user'
              AND m.content IS NOT NULL AND length(m.content) > 25
              AND m.active = 1
              AND s.source NOT IN ('cron', 'subagent', 'tool')
              AND m.timestamp > ?
            ORDER BY m.timestamp DESC
            LIMIT ?
            """, (since, int(limit)))
        out = []
        for content, source, ts in cur.fetchall():
            text = str(content).strip()
            if text.startswith("/") or text.startswith("[mirror]"):
                continue
            out.append({"content": text[:300], "source": source, "ts": ts})
        return out
    finally:
        conn.close()


def mine(ctx, store: SuggestionStore, days: int = 14) -> Dict[str, Any]:
    """Run one mining pass. Returns {added, considered, error?}."""
    messages = harvest_user_messages(days=days)
    if len(messages) < 12:
        return {"added": 0, "considered": 0,
                "note": f"only {len(messages)} interactive messages in window — not enough signal"}

    numbered = "\n".join(
        f"{i+1}. [{m['source']}, {time.strftime('%m-%d %H:%M', time.localtime(m['ts']))}] {m['content']}"
        for i, m in enumerate(messages))
    try:
        result = ctx.llm.complete_structured(
            instructions=_MINE_INSTRUCTIONS,
            input=[numbered],
            json_schema=_SUGGESTION_SCHEMA,
            schema_name="autopilot_patterns",
            purpose="autopilot-mine",
        )
        data = result.data if hasattr(result, "data") else {}
    except Exception as exc:
        logger.debug("autopilot: mine LLM call failed: %s", exc)
        return {"added": 0, "considered": len(messages), "error": str(exc)}

    patterns = (data or {}).get("patterns") or []
    existing = store.suggestions()
    known_fp = {s.get("fingerprint") for s in existing}
    known_fp |= {s.get("fingerprint") for s in existing if s.get("status") == "dismissed"}
    cron_names = _existing_cron_names()
    added = 0
    for p in patterns:
        try:
            if float(p.get("confidence") or 0) < 0.6 or int(p.get("approx_count") or 0) < 3:
                continue
        except Exception:
            continue
        title = str(p.get("title") or "").strip()
        if not title or title.lower() in cron_names:
            continue
        fp = _fingerprint(title + " " + str(p.get("draft_prompt", ""))[:120])
        if fp in known_fp:
            continue
        existing.append({
            "id": fp,
            "fingerprint": fp,
            "status": "pending",
            "created_at": time.time(),
            **{k: p.get(k) for k in ("title", "rationale", "examples", "approx_count",
                                     "kind", "schedule", "draft_prompt", "confidence")},
        })
        known_fp.add(fp)
        added += 1
    store.save_suggestions(existing[-60:])
    store.save_state({**store.state(), "last_mine": time.time(),
                      "last_window_days": days})
    return {"added": added, "considered": len(messages),
            "patterns_found": len(patterns)}


def approve(ctx, store: SuggestionStore, sug_id: str) -> Dict[str, Any]:
    sug = store.get(sug_id)
    if not sug:
        return {"success": False, "error": f"no suggestion {sug_id}"}
    if sug.get("status") == "approved":
        return {"success": False, "error": "already approved"}
    kind = sug.get("kind") or "cron"
    if kind == "memory":
        try:
            from hermes_constants import get_hermes_home

            mem = Path(get_hermes_home()) / "MEMORY.md"
            with open(mem, "a", encoding="utf-8") as f:
                f.write(f"\n- {sug['draft_prompt']}  (pinned by autopilot, "
                        f"{time.strftime('%Y-%m-%d')})\n")
            store.update(sug_id, status="approved", approved_at=time.time())
            return {"success": True, "kind": "memory", "note": "pinned to MEMORY.md"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}
    if kind != "cron":
        return {"success": False,
                "error": f"auto-approval for kind={kind} not supported — ask the agent to create it from the draft"}
    try:
        from cron.jobs import create_job

        job = create_job(
            prompt=str(sug["draft_prompt"]),
            schedule=str(sug.get("schedule") or "every day 09:00"),
            name=str(sug["title"])[:60],
            deliver="local",
        )
        store.update(sug_id, status="approved", approved_at=time.time(),
                     job_id=job.get("id") or job.get("job_id"))
        return {"success": True, "kind": "cron",
                "job": {k: job.get(k) for k in ("id", "job_id", "name", "schedule")
                        if job.get(k)}}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ------------------------------------------------------------ background

class AutoMiner:
    """Debounced background mining triggered by on_session_end."""

    def __init__(self, ctx, store: SuggestionStore):
        self.ctx = ctx
        self.store = store
        self._lock = threading.Lock()
        self._running = False

    def note_session_end(self) -> None:
        pending = self.store.pending()
        st = self.store.state()
        if time.time() - float(st.get("last_mine") or 0) < _COOLDOWN_S:
            self.store.save_pending(pending)  # still record, mine later
            return
        if len(pending) + 1 < _MINE_AFTER_SESSIONS:
            self.store.save_pending(pending)
            return
        self._kick()

    def _kick(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True

        def _run():
            try:
                mine(self.ctx, self.store)
                self.store.save_pending([])
            except Exception as exc:
                logger.debug("autopilot: background mine failed: %s", exc)
            finally:
                with self._lock:
                    self._running = False

        threading.Thread(target=_run, name="autopilot-miner", daemon=True).start()
