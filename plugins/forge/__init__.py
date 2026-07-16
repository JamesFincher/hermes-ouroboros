"""Forge — personal self-distillation for Hermes.

Your daily Hermes usage is the best possible fine-tuning corpus: real
tasks, in your environment, with the corrections you already taught the
agent.  Forge harvests completed interactive sessions, filters them with
free heuristics, scores the survivors with a rubric LLM pass
(ctx.llm.complete_structured — host-owned, no extra keys), and appends
the keepers to a Hermes-format ShareGPT dataset at
~/.hermes/forge/dataset.jsonl.

Tool:
    forge_harvest(lookback_days, max_sessions, min_score)
    — the agent can run a harvest conversationally ("forge this month's
      sessions").

Commands:
    /forge harvest [days]   — run a harvest now
    /forge stats            — dataset size, task-type mix
    /forge export <path>    — copy JSONL + dataset card somewhere
    /forge tail             — most recent kept sessions

Automate it (weekly distillation):
    hermes cron create "every sunday 03:00" --prompt "Run forge_harvest \\
      with lookback_days 7 and report what was kept."
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CTX = None


def _forge_dir() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "forge"


# ------------------------------------------------------------------- tool

_SCHEMA = {
    "name": "forge_harvest",
    "description": (
        "Harvest your own past Hermes sessions into a fine-tuning dataset. "
        "Selects completed interactive sessions with real tool use, "
        "heuristically filters out frustrating/failed/looping sessions, "
        "rubric-scores the survivors, and appends the best to a local "
        "ShareGPT-format JSONL dataset (deduped across runs). Use when the "
        "user asks to distill, harvest, or build training data from their "
        "usage. Read-only against the session database."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "lookback_days": {"type": "integer",
                              "description": "How far back to mine (default 30)."},
            "max_sessions": {"type": "integer",
                             "description": "Cap on sessions to pre-score (default 20)."},
            "min_score": {"type": "number",
                          "description": "Rubric threshold 0-1 to keep a session "
                                         "(default 0.7)."},
        },
    },
}


def _handle_harvest(params, **kwargs):
    from .harvest import harvest

    result = harvest(
        _CTX,
        _forge_dir(),
        days=int((params or {}).get("lookback_days") or 30),
        max_sessions=int((params or {}).get("max_sessions") or 20),
        min_score=float((params or {}).get("min_score") or 0.7),
    )
    return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- command

def _cmd_forge(raw_args: str) -> str:
    from .harvest import dataset_stats, export, harvest

    parts = (raw_args or "").strip().split()
    sub = parts[0].lower() if parts else "stats"

    if sub == "harvest":
        days = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 30
        result = harvest(_CTX, _forge_dir(), days=days)
        if not result.get("success"):
            return f"Harvest failed: {result.get('error')}"
        kept = result["kept"]
        lines = [f"Harvest: {result['candidates']} candidates → "
                 f"{result['prescored']} pre-scored → {result['rubric_scored']} "
                 f"rubric-scored → {len(kept)} kept. "
                 f"Dataset now {result['dataset_size']} entries "
                 f"({result['dataset']})."]
        for k in kept[:8]:
            lines.append(f"  + [{k['score']:.2f}] {k['task_type']}: {k['reason'][:90]}")
        return "\n".join(lines)

    if sub == "stats":
        st = dataset_stats(_forge_dir())
        return (f"Forge dataset: {st['entries']} entries, "
                f"{st['dataset_bytes'] / 1e6:.2f} MB\n"
                f"Path: {st['dataset_path']}\n"
                f"Task mix: {json.dumps(st['by_task_type'], indent=2)}")

    if sub == "export":
        if len(parts) < 2:
            return "Usage: /forge export <destination-dir>"
        result = export(_forge_dir(), Path(parts[1]))
        return (f"Exported {result['entries']} entries to {result['exported_to']} "
                "(hermes_forge_dataset.jsonl + DATASET_CARD.md)."
                if result.get("success") else f"Export failed: {result.get('error')}")

    if sub == "tail":
        index_path = _forge_dir() / "index.json"
        if not index_path.exists():
            return "No forge dataset yet — run /forge harvest."
        index = json.loads(index_path.read_text(encoding="utf-8"))
        recent = sorted(index.items(), key=lambda kv: kv[1].get("kept_at", 0),
                        reverse=True)[:10]
        lines = ["Recently distilled sessions:"]
        for fp, v in recent:
            when = time.strftime("%m-%d %H:%M", time.localtime(v.get("kept_at", 0)))
            lines.append(f"  {when}  [{v.get('score', 0):.2f}] "
                         f"{v.get('task_type', '?')}  session {v.get('session_id', '')[:8]}…")
        return "\n".join(lines)

    return "Usage: /forge [harvest [days]|stats|export <path>|tail]"


# --------------------------------------------------------------- register

def register(ctx):
    global _CTX
    _CTX = ctx
    ctx.register_tool(
        name="forge_harvest",
        toolset="forge",
        schema=_SCHEMA,
        handler=_handle_harvest,
        description="Distill past sessions into a ShareGPT fine-tuning dataset.",
        emoji="⚒️",
    )
    ctx.register_command(
        "forge", _cmd_forge,
        description="Self-distillation: harvest, stats, export, tail",
        args_hint="[harvest [days]|stats|export <path>|tail]",
    )
