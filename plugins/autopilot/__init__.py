"""Autopilot — turns your repeated requests into scheduled automations.

Hermes already knows *what you did* (session insights) and *how to
schedule things* (cron).  Autopilot closes the loop between them: it
mines the local session DB for recurring intents, drafts self-contained
cron jobs (or skill outlines / memory pins) with a structured LLM pass,
and lets you approve them with one command.

Flow:
    You chat as usual. Every ~5 completed sessions (max once per 6h),
    Autopilot re-mines the last 14 days of your interactive messages.
    /autopilot list            — pending suggestions
    /autopilot show <id>       — evidence + full draft prompt
    /autopilot approve <id>    — create the cron job (or pin the memory)
    /autopilot dismiss <id>    — never suggest this again
    /autopilot scan [days]     — force a mining pass right now
    /autopilot stats           — mining state and counts

The model can also drive it conversationally via the autopilot_scan tool
("what could you automate for me?").
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_store = None
_miner = None
_CTX = None


def _ensure(ctx=None):
    global _store, _miner, _CTX
    if ctx is not None:
        _CTX = ctx
    if _store is None:
        from hermes_constants import get_hermes_home

        from .miner import AutoMiner, SuggestionStore

        _store = SuggestionStore(Path(get_hermes_home()) / "autopilot")
        _miner = AutoMiner(_CTX, _store)
    return _store, _miner


# ------------------------------------------------------------------- hook

def _on_session_end(session_id: str = "", completed: bool = False,
                    platform: str = "", **kwargs):
    try:
        store, miner = _ensure()
        if not completed:
            return
        pending = store.pending()
        if session_id and session_id not in pending:
            pending.append(session_id)
            store.save_pending(pending)
        miner.note_session_end()
    except Exception as exc:
        logger.debug("autopilot: on_session_end failed: %s", exc)


# ------------------------------------------------------------------- tool

_SCAN_SCHEMA = {
    "name": "autopilot_scan",
    "description": (
        "Scan recent conversation history for recurring user requests that "
        "could be automated, and return concrete automation suggestions "
        "(cron job drafts, skill outlines, memory pins). Use when the user "
        "asks what you could automate, or proactively after noticing the "
        "same request several times. Suggestions are queued for review "
        "with /autopilot — nothing is created without user approval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "lookback_days": {"type": "integer",
                              "description": "Days of history to mine (default 14)."},
        },
    },
}


def _handle_scan(params, **kwargs):
    from .miner import mine

    store, _ = _ensure()
    days = int((params or {}).get("lookback_days") or 14)
    result = mine(_CTX, store, days=days)
    pending = [s for s in store.suggestions() if s.get("status") == "pending"]
    return json.dumps({
        "success": True,
        "scan": result,
        "pending_suggestions": [
            {"id": s["id"], "title": s["title"], "kind": s["kind"],
             "schedule": s.get("schedule"), "confidence": s.get("confidence"),
             "rationale": s.get("rationale")}
            for s in pending[-10:]
        ],
        "note": "User reviews with /autopilot list and approves with "
                "/autopilot approve <id>. Never auto-create jobs.",
    }, ensure_ascii=False)


# ---------------------------------------------------------------- command

def _fmt_suggestion(s, verbose=False):
    conf = s.get("confidence")
    line = (f"[{s['id']}] ({s.get('kind')}, conf {conf:.2f}) {s.get('title')}"
            if isinstance(conf, (int, float)) else f"[{s['id']}] {s.get('title')}")
    if s.get("schedule"):
        line += f"  — {s['schedule']}"
    if verbose:
        line += (f"\n    why: {s.get('rationale')}"
                 f"\n    seen ~{s.get('approx_count')}x, e.g.: "
                 + " | ".join((s.get("examples") or [])[:2])
                 + f"\n    draft: {s.get('draft_prompt')}")
    return line


def _cmd_autopilot(raw_args: str) -> str:
    from .miner import approve, mine

    store, _ = _ensure()
    parts = (raw_args or "").strip().split()
    sub = parts[0].lower() if parts else "list"

    if sub == "list":
        pending = [s for s in store.suggestions() if s.get("status") == "pending"]
        if not pending:
            return ("No pending suggestions. Autopilot mines automatically "
                    "every few sessions, or run /autopilot scan now.")
        return "Pending automations:\n" + "\n".join(
            _fmt_suggestion(s) for s in pending[-15:]) + \
            "\n\n/autopilot show <id> for details, /autopilot approve <id> to create."

    if sub == "show" and len(parts) > 1:
        sug = store.get(parts[1])
        return _fmt_suggestion(sug, verbose=True) if sug else f"No suggestion {parts[1]}."

    if sub == "approve" and len(parts) > 1:
        result = approve(_CTX, store, parts[1])
        if result.get("success"):
            if result.get("kind") == "cron":
                return (f"Created cron job {result.get('job')}. "
                        "Manage with `hermes cron list` / `/cron`.")
            return result.get("note", "Done.")
        return f"Approve failed: {result.get('error')}"

    if sub == "dismiss" and len(parts) > 1:
        return ("Dismissed — won't suggest again." if store.update(parts[1], status="dismissed")
                else f"No suggestion {parts[1]}.")

    if sub == "scan":
        days = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 14
        result = mine(_CTX, store, days=days)
        return (f"Scan complete: {result.get('added', 0)} new suggestions "
                f"({result.get('patterns_found', 0)} patterns in "
                f"{result.get('considered', 0)} messages). /autopilot list to review."
                + (f"\n(note: {result['note']})" if result.get("note") else "")
                + (f"\n(error: {result['error']})" if result.get("error") else ""))

    if sub == "stats":
        items = store.suggestions()
        st = store.state()
        by_status = {}
        for s in items:
            by_status[s.get("status", "?")] = by_status.get(s.get("status", "?"), 0) + 1
        last = st.get("last_mine")
        return (f"Autopilot: {json.dumps(by_status)} suggestions; "
                f"last mine: {time.strftime('%Y-%m-%d %H:%M', time.localtime(last)) if last else 'never'}; "
                f"pending sessions queued: {len(store.pending())}")

    return ("Usage: /autopilot [list|show <id>|approve <id>|dismiss <id>|"
            "scan [days]|stats]")


# --------------------------------------------------------------- register

def register(ctx):
    _ensure(ctx)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_tool(
        name="autopilot_scan",
        toolset="autopilot",
        schema=_SCAN_SCHEMA,
        handler=_handle_scan,
        description="Detect recurring user requests and draft automations.",
        emoji="🛩️",
    )
    ctx.register_command(
        "autopilot", _cmd_autopilot,
        description="Habit-to-automation miner: list/approve/dismiss/scan",
        args_hint="[list|show|approve|dismiss|scan|stats]",
    )
