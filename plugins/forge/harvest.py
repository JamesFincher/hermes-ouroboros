"""Forge harvester — turn organic Hermes usage into training data.

Hermes can already *generate* trajectories (batch_runner.py runs synthetic
prompt lists and saves ShareGPT JSONL).  Forge is the other direction: it
*harvests* the sessions you actually had — real tasks, real tool chains,
real recoveries from real errors — and distills them into the same
Hermes trajectory format.

Selection pipeline (all read-only against state.db):

  1. Candidates: completed interactive sessions with real tool use
     (message_count >= 8, tool_call_count >= 2), excluding
     cron/subagent/tool sessions.
  2. Heuristic pre-score (free): tool error ratio, user-frustration
     markers, session abandonment, length.  Top N survive.
  3. Rubric LLM pass (ctx.llm.complete_structured): is this transcript a
     good *teaching* example?  Keeps score >= threshold.
  4. Convert to Hermes' trajectory shape (system + human/gpt with
     <tool_call>/<tool_response> blocks, reasoning in <think> tags),
     dedupe by content fingerprint, append to ~/.hermes/forge/dataset.jsonl.

The point: fine-tuning on your own corrected workflows is how an agent
that "grows with you" literally grows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_FRUSTRATION_RE = re.compile(
    r"(that'?s (wrong|not right)|not what i (asked|meant)|try again|you broke|"
    r"still (failing|broken|wrong)|doesn'?t work|stop doing|why did you)",
    re.IGNORECASE)

_RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "overall": {"type": "number"},
        "task_type": {"type": "string"},
        "teaches_reusable_pattern": {"type": "boolean"},
        "has_clean_completion": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["overall", "task_type", "teaches_reusable_pattern", "reason"],
}

_RUBRIC_INSTRUCTIONS = """You are curating training data for a tool-using AI agent. You receive a digest of one real agent session: the opening user request, the sequence of tools used, whether tool calls failed, and the final assistant answer.

Score the session as a TRAINING EXAMPLE:
- overall (0-1): would fine-tuning on this transcript make the model better at similar tasks? High scores need: a clear task, a sensible tool chain, visible progress, and a correct, complete final answer. Penalize: confusion loops, user frustration, abandoned tasks, trivial chit-chat, sensitive data (credentials, personal secrets — flag with overall <= 0.2).
- task_type: short label (e.g. "bug-fix", "data-pipeline", "research", "devops").
- teaches_reusable_pattern: does the tool sequence generalize beyond this one instance?
- has_clean_completion: did the session actually finish the task?
- reason: one sentence."""


# ------------------------------------------------------------------ db

def _connect_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    conn.row_factory = sqlite3.Row
    return conn


def _candidate_sessions(conn: sqlite3.Connection, days: int,
                        limit: int = 400) -> List[Dict[str, Any]]:
    since = time.time() - days * 86400
    cur = conn.execute(
        """
        SELECT id, source, started_at, ended_at, end_reason, message_count,
               tool_call_count, model, title, system_prompt
        FROM sessions
        WHERE ended_at IS NOT NULL
          AND COALESCE(archived, 0) = 0
          AND source NOT IN ('cron', 'subagent', 'tool')
          AND message_count >= 8
          AND tool_call_count >= 2
          AND started_at > ?
        ORDER BY started_at DESC
        LIMIT ?
        """, (since, int(limit)))
    return [dict(r) for r in cur.fetchall()]


def _session_messages(conn: sqlite3.Connection, session_id: str) -> List[Dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT role, content, tool_calls, tool_name, reasoning, timestamp
        FROM messages
        WHERE session_id = ? AND COALESCE(active, 1) = 1
        ORDER BY timestamp, id
        """, (session_id,))
    return [dict(r) for r in cur.fetchall()]


# ------------------------------------------------------------- heuristics

def _prescore(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    user_msgs = [m for m in messages if m["role"] == "user"]
    tool_errors = 0
    for m in tool_msgs:
        head = str(m.get("content") or "")[:300].lower()
        if '"success": false' in head or '"error"' in head or head.startswith("error"):
            tool_errors += 1
    frustration = sum(1 for m in user_msgs
                      if _FRUSTRATION_RE.search(str(m.get("content") or "")))
    error_ratio = tool_errors / max(1, len(tool_msgs))
    last = messages[-1] if messages else {}
    clean_end = last.get("role") == "assistant" and bool(str(last.get("content") or "").strip())
    score = 1.0
    score -= 0.6 * error_ratio
    score -= 0.25 * min(frustration, 3)
    if not clean_end:
        score -= 0.3
    if len(messages) > 200:
        score -= 0.1  # meandering marathons teach loops, not patterns
    return {"score": max(0.0, score), "tool_errors": tool_errors,
            "tool_calls": len(tool_msgs), "frustration": frustration,
            "clean_end": clean_end}


def _digest(messages: List[Dict[str, Any]], max_total: int = 6000) -> str:
    parts = []
    total = 0
    first_user = next((m for m in messages if m["role"] == "user"), None)
    if first_user:
        parts.append("OPENING REQUEST:\n" + str(first_user.get("content") or "")[:800])
    tool_seq = [m.get("tool_name") or "?" for m in messages if m["role"] == "tool"]
    parts.append("TOOL SEQUENCE (" + str(len(tool_seq)) + "): "
                 + " → ".join(tool_seq[:60]))
    final = next((m for m in reversed(messages)
                  if m["role"] == "assistant" and str(m.get("content") or "").strip()), None)
    if final:
        parts.append("FINAL ANSWER:\n" + str(final.get("content") or "")[:1200])
    text = "\n\n".join(parts)
    return text[:max_total]


# ------------------------------------------------------------ conversion

def _to_trajectory(system_prompt: str, messages: List[Dict[str, Any]],
                   max_system_chars: int = 4000) -> List[Dict[str, str]]:
    """Hermes trajectory format: system + human/gpt turns, tool calls in
    <tool_call> blocks, tool results as role 'tool' in <tool_response>
    blocks, reasoning wrapped in <think>.  Mirrors
    agent.agent_runtime_helpers.convert_to_trajectory_format."""
    traj: List[Dict[str, str]] = []
    sys_msg = (system_prompt or
               "You are a function calling AI model. You are provided with "
               "function signatures within <tools> </tools> XML tags.")
    traj.append({"from": "system", "value": sys_msg[:max_system_chars]})
    skipped_first_user = False
    for m in messages:
        role = m["role"]
        content = str(m.get("content") or "")
        if role == "system":
            continue
        if role == "user":
            traj.append({"from": "human", "value": content})
        elif role == "assistant":
            value = ""
            reasoning = str(m.get("reasoning") or "").strip()
            if reasoning:
                value += f"<think>\n{reasoning}\n</think>\n"
            if content.strip():
                value += content + "\n"
            tool_calls = m.get("tool_calls")
            if tool_calls:
                try:
                    calls = json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls
                except Exception:
                    calls = []
                for call in calls or []:
                    if not isinstance(call, dict):
                        continue
                    fn = (call.get("function") or {})
                    name = fn.get("name") or call.get("name") or "?"
                    args = fn.get("arguments") or call.get("arguments") or {}
                    if isinstance(args, str):
                        arg_str = args
                    else:
                        arg_str = json.dumps(args, ensure_ascii=False)
                    value += f"<tool_call>\n{{'name': '{name}', 'arguments': {arg_str}}}\n</tool_call>\n"
            if value.strip():
                traj.append({"from": "gpt", "value": value.rstrip()})
        elif role == "tool":
            traj.append({"from": "tool",
                         "value": f"<tool_response>\n{content[:6000]}\n</tool_response>"})
    # need at least one human and one gpt turn
    roles = {t["from"] for t in traj}
    if "human" not in roles or "gpt" not in roles:
        return []
    return traj


def _fingerprint(traj: List[Dict[str, str]]) -> str:
    first_human = next((t["value"] for t in traj if t["from"] == "human"), "")
    tools = [t["value"][:60] for t in traj if t["from"] == "tool"]
    blob = first_human[:500] + "|" + "|".join(tools[:30])
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:16]


# --------------------------------------------------------------- harvest

def harvest(ctx, forge_dir: Path, days: int = 30, max_sessions: int = 20,
            min_score: float = 0.7, max_llm: int = 12) -> Dict[str, Any]:
    from hermes_constants import get_hermes_home

    forge_dir = Path(forge_dir)
    forge_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(get_hermes_home()) / "state.db"
    if not db_path.exists():
        return {"success": False, "error": f"no session db at {db_path}"}

    index_path = forge_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    dataset_path = forge_dir / "dataset.jsonl"

    conn = _connect_ro(db_path)
    try:
        candidates = _candidate_sessions(conn, days)
        pres = []
        for s in candidates:
            if s["id"] in index:
                continue
            msgs = _session_messages(conn, s["id"])
            if len(msgs) < 8:
                continue
            heur = _prescore(msgs)
            pres.append((heur["score"], s, msgs, heur))
        pres.sort(key=lambda t: t[0], reverse=True)
        pres = pres[:max_sessions]
    finally:
        conn.close()

    kept, scored = [], 0
    for heur_score, s, msgs, heur in pres:
        if scored >= max_llm:
            break
        scored += 1
        try:
            result = ctx.llm.complete_structured(
                instructions=_RUBRIC_INSTRUCTIONS,
                input=[_digest(msgs)],
                json_schema=_RUBRIC_SCHEMA,
                schema_name="forge_rubric",
                purpose="forge-rubric",
            )
            rubric = result.data if hasattr(result, "data") else {}
        except Exception as exc:
            logger.debug("forge: rubric failed for %s: %s", s["id"], exc)
            continue
        overall = float((rubric or {}).get("overall") or 0)
        if overall < min_score:
            continue
        traj = _to_trajectory(s.get("system_prompt") or "", msgs)
        if not traj:
            continue
        fp = _fingerprint(traj)
        if fp in index:
            continue
        entry = {
            "conversations": traj,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": s.get("model") or "",
            "completed": True,
            "forge": {
                "session_id": s["id"],
                "task_type": rubric.get("task_type"),
                "rubric_score": overall,
                "heuristic": heur,
                "fingerprint": fp,
                "source": s.get("source"),
                "session_started": s.get("started_at"),
            },
        }
        with open(dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        index[fp] = {"session_id": s["id"], "kept_at": time.time(),
                     "score": overall, "task_type": rubric.get("task_type")}
        kept.append({"session_id": s["id"], "task_type": rubric.get("task_type"),
                     "score": overall, "reason": rubric.get("reason")})

    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return {
        "success": True,
        "candidates": len(candidates),
        "prescored": len(pres),
        "rubric_scored": scored,
        "kept": kept,
        "dataset": str(dataset_path),
        "dataset_size": len(index),
    }


def dataset_stats(forge_dir: Path) -> Dict[str, Any]:
    forge_dir = Path(forge_dir)
    index_path = forge_dir / "index.json"
    dataset_path = forge_dir / "dataset.jsonl"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    by_type: Dict[str, int] = {}
    for v in index.values():
        t = v.get("task_type") or "unlabeled"
        by_type[t] = by_type.get(t, 0) + 1
    return {
        "entries": len(index),
        "by_task_type": by_type,
        "dataset_bytes": dataset_path.stat().st_size if dataset_path.exists() else 0,
        "dataset_path": str(dataset_path),
    }


def export(forge_dir: Path, dest: Path) -> Dict[str, Any]:
    import shutil

    forge_dir = Path(forge_dir)
    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    dataset = forge_dir / "dataset.jsonl"
    if not dataset.exists():
        return {"success": False, "error": "no dataset yet — run a harvest first"}
    shutil.copyfile(dataset, dest / "hermes_forge_dataset.jsonl")
    stats = dataset_stats(forge_dir)
    card = (
        "# Hermes Forge Dataset\n\n"
        f"Exported {time.strftime('%Y-%m-%d %H:%M')}.\n\n"
        f"- Entries: {stats['entries']}\n"
        f"- Format: Hermes trajectory / ShareGPT-style JSONL "
        f"(`conversations`: system/human/gpt/tool, tool calls in "
        f"`<tool_call>` blocks, results in `<tool_response>` blocks, "
        f"reasoning in `<think>` tags)\n"
        f"- By task type: {json.dumps(stats['by_task_type'], indent=2)}\n\n"
        "Each entry carries a `forge` metadata block with the source "
        "session id, rubric score and task type. Sessions were "
        "auto-selected from completed interactive use and rubric-scored; "
        "review for sensitive content before training.\n")
    (dest / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    return {"success": True, "exported_to": str(dest), **stats}
