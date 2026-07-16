"""Echo — proactive cross-session recall for Hermes.

Every turn, before the model sees your message, Echo searches your own
past sessions in the local SQLite session store (FTS5) and injects the
most relevant excerpts as context.  No external service, no embeddings,
no configuration: enable it and your agent starts remembering what you
did last Tuesday.

Deep systems used:
  - ``pre_llm_call`` hook with ``{"context": ...}`` injection (the channel
    Hermes reserves for per-turn context plugins)
  - ``hermes_state.SessionDB`` read-only attach + ``search_messages`` FTS5
    (with trigram/CJK fallback inherited from the host)
  - ``on_session_start`` hook for per-conversation anti-spam state

Tools:
    echo_recall(query, limit, window) — explicit deep recall; returns
    ranked sessions with ±window message context around each hit.

Commands:
    /echo status   — enabled state, config, last injected block
    /echo on|off   — toggle auto-injection (persisted)
    /echo terms <text> — preview the query terms Echo extracts
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from hermes_constants import get_hermes_home

        from .recall import RecallEngine

        base = Path(get_hermes_home()) / "echo"
        base.mkdir(parents=True, exist_ok=True)
        _engine = RecallEngine(base / "config.json")
    return _engine


# ------------------------------------------------------------------ hooks

def _pre_llm_call(user_message: str = "", session_id: str = "", **kwargs):
    try:
        block = _get_engine().build_turn_context(user_message, session_id)
        if block:
            return {"context": block}
    except Exception as exc:  # never break a turn
        logger.debug("echo: pre_llm_call failed: %s", exc)
    return None


def _on_session_start(session_id: str = "", **kwargs):
    try:
        if session_id:
            _get_engine().reset_session(session_id)
    except Exception as exc:
        logger.debug("echo: on_session_start failed: %s", exc)


# ------------------------------------------------------------------- tool

_SCHEMA = {
    "name": "echo_recall",
    "description": (
        "Deep-recall your own past conversations. Searches every past "
        "Hermes session (CLI, Telegram, cron, …) full-text and returns the "
        "most relevant sessions with message context around each hit. Use "
        "when the user asks 'when did we…', 'what did we decide about…', or "
        "when automatically-injected recall points at a session worth "
        "opening up. Read-only; never modifies history."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to recall — keywords or a phrase.",
            },
            "limit": {
                "type": "integer",
                "description": "Max sessions to return (default 5).",
            },
        },
        "required": ["query"],
    },
}


def _handle_recall(params, **kwargs):
    query = (params or {}).get("query", "")
    limit = int((params or {}).get("limit") or 5)
    engine = _get_engine()
    hits = engine.recall(query, limit=limit)
    if not hits:
        return json.dumps({"success": True, "hits": [],
                           "note": "No past sessions matched."})
    return json.dumps({
        "success": True,
        "terms": hits[0]["terms"],
        "hits": [
            {
                "session_id": h["session_id"],
                "when": h["session_started"],
                "source": h["source"],
                "role": h["role"],
                "snippet": h["snippet"],
                "nearby": h["context"],
                "hint": ("Open the full window with session_search "
                         f"(session_id={h['session_id'][:8]}…)."),
            }
            for h in hits
        ],
    }, ensure_ascii=False)


# ---------------------------------------------------------------- command

def _cmd_echo(raw_args: str) -> str:
    engine = _get_engine()
    parts = (raw_args or "").strip().split(maxsplit=1)
    sub = parts[0].lower() if parts and parts[0] else "status"

    if sub == "on":
        engine.config["enabled"] = True
        engine.save_config()
        return "Echo auto-recall enabled."
    if sub == "off":
        engine.config["enabled"] = False
        engine.save_config()
        return "Echo auto-recall disabled (echo_recall tool still works)."

    if sub == "terms" and len(parts) > 1:
        from .recall import extract_query_terms, build_fts_query

        terms = extract_query_terms(parts[1])
        return (f"Terms: {terms}\nFTS query: {build_fts_query(terms)}"
                if terms else "No usable terms in that text.")

    if sub == "status":
        cfg = {k: engine.config[k] for k in sorted(engine.config)}
        last = engine.last_block or "(nothing injected yet this process)"
        return (f"Echo config: {json.dumps(cfg)}\n\nLast injected block:\n{last}")

    return "Usage: /echo [status|on|off|terms <text>]"


# --------------------------------------------------------------- register

def register(ctx):
    ctx.register_hook("pre_llm_call", _pre_llm_call)
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_tool(
        name="echo_recall",
        toolset="echo",
        schema=_SCHEMA,
        handler=_handle_recall,
        description="Recall relevant excerpts from your own past sessions.",
        emoji="🔁",
    )
    ctx.register_command(
        "echo", _cmd_echo,
        description="Proactive recall: status, on/off, term preview",
        args_hint="[status|on|off|terms <text>]",
    )
