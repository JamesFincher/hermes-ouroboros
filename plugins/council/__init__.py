"""Council — adversarial parallel delegation for Hermes.

When a question matters more than it costs, don't ask one agent — convene
a council.  Each councillor is a full tool-using subagent (via the
delegate_task registry, so concurrency limits, spawn depth and cost
roll-up all apply), briefed with an adversarial lens; a host-owned judge
(ctx.llm structured completion — no extra API keys) scores, synthesizes,
and preserves dissent.

Tool:
    council_convene(question, context, mode, councilors)
    modes: deliberate | redteam | race

Command:
    /council <question>            deliberate mode, 3 members
    /council redteam <proposal>    red-team a plan
    /council race <task> [n]       n competing solution attempts

Config (optional — fail-closed trust gate applies):
    plugins.entries.council.judge_model: "openrouter/some-cheap-model"
    plugins.entries.council.llm.allow_model_override: true
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_SCHEMA = {
    "name": "council_convene",
    "description": (
        "Convene an adversarial council for a hard question. Fans the "
        "question out to 2-5 parallel subagents with distinct lenses "
        "(analyst/skeptic/pragmatist/contrarian/user-advocate), then a "
        "judge pass scores each position, synthesizes the strongest "
        "answer, and preserves unresolved dissent. Use for architecture "
        "decisions, tricky debugging theories, design trade-offs, plans "
        "worth attacking before executing. Expensive by design — do not "
        "use for routine tasks. Modes: deliberate (default), redteam "
        "(attack a proposal), race (competing solutions)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string",
                         "description": "The question or proposal to deliberate. "
                                        "Self-contained — councillors do not see "
                                        "this conversation."},
            "context": {"type": "string",
                        "description": "Optional shared evidence: file paths, "
                                       "error logs, constraints."},
            "mode": {"type": "string", "enum": ["deliberate", "redteam", "race"],
                     "description": "deliberate = lenses on one question; "
                                    "redteam = architect + attackers; "
                                    "race = competing solutions. Default deliberate."},
            "councilors": {"type": "integer",
                           "description": "Council size 2-5 (default 3). "
                                          "Each member is a full subagent."},
        },
        "required": ["question"],
    },
}


def _judge_model() -> str:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        entry = ((cfg.get("plugins") or {}).get("entries") or {}).get("council") or {}
        return str(entry.get("judge_model") or "")
    except Exception:
        return ""


def _handle_convene(params, **kwargs):
    from .council import convene

    result = convene(
        _CTX,
        question=str((params or {}).get("question") or ""),
        context=str((params or {}).get("context") or ""),
        mode=str((params or {}).get("mode") or "deliberate"),
        councilors=int((params or {}).get("councilors") or 3),
        parent_agent=kwargs.get("parent_agent"),
        judge_model=_judge_model(),
    )
    return json.dumps(result, ensure_ascii=False, default=str)


_CTX = None  # bound at register() time; used by the tool handler


def _cmd_council(raw_args: str) -> str:
    from .council import convene, format_markdown

    text = (raw_args or "").strip()
    if not text:
        return ("Usage: /council [redteam|race] <question>\n"
                "  /council Should we migrate state.db to Postgres?\n"
                "  /council redteam Here is my deployment plan: …\n"
                "  /council race Write a rate limiter for the gateway")
    mode = "deliberate"
    for m in ("redteam", "race", "deliberate"):
        if text.lower().startswith(m + " "):
            mode = m
            text = text[len(m):].strip()
            break
    result = convene(_CTX, question=text, mode=mode, councilors=3,
                     judge_model=_judge_model())
    return format_markdown(result)


def register(ctx):
    global _CTX
    _CTX = ctx
    ctx.register_tool(
        name="council_convene",
        toolset="council",
        schema=_SCHEMA,
        handler=_handle_convene,
        description="Adversarial multi-subagent deliberation with a judged synthesis.",
        emoji="🏛️",
    )
    ctx.register_command(
        "council", _cmd_council,
        description="Convene an adversarial council (deliberate|redteam|race)",
        args_hint="[redteam|race] <question>",
    )
