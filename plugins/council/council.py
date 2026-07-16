"""Council engine — adversarial multi-subagent deliberation.

Hermes has two adjacent primitives, neither of which does this:

  * delegate_task — spawns tool-using subagents in isolated contexts
    (built for *dividing* work, not *arguing about* it)
  * Mixture-of-Agents provider — tool-less reference models advise the
    aggregator inside one normal turn

Council composes them conceptually but goes further: each councillor is a
FULL subagent with tool access (it can read the repo, run commands, check
docs before taking a position), lenses are adversarial by design, and a
host-owned judge pass (ctx.llm.complete_structured — the plugin LLM
facade Hermes added in v0.14) scores positions, synthesizes, and —
critically — returns the dissent instead of averaging it away.

Modes:
  deliberate — same question, N lenses; judge synthesizes.
  redteam    — one architect proposes, N-1 attackers try to break it,
               judge returns verdict + must-fix list.
  race       — N independent solution attempts with different
               optimization briefs; judge ranks them, keeps the best two.

Judge model routing: the judge runs on the user's active model through
ctx.llm (no extra keys needed).  To pin a cheaper judge, set
plugins.entries.council.judge_model in config.yaml AND allow the override:
plugins.entries.council.llm.allow_model_override: true — Hermes' trust
policy is fail-closed, so without that flag the pin is ignored and the
active model judges.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LENSES = [
    ("Analyst", "Decompose the problem. Evaluate the realistic options "
     "systematically, ground every claim in something checkable, and give "
     "a clear recommendation with trade-offs."),
    ("Skeptic", "Your job is to be wrong on purpose if necessary: attack "
     "the obvious answer. Find failure modes, hidden assumptions, edge "
     "cases, and what breaks at scale or in six months."),
    ("Pragmatist", "Optimize for the cheapest robust path. 80/20 "
     "solutions, operational reality, maintenance burden, time-to-value. "
     "Reject elegance that costs weeks."),
    ("Contrarian", "Argue for the approach everyone else will likely "
     "dismiss. If consensus forms around X, steelman not-X. You win by "
     "finding the option nobody evaluated."),
    ("UserAdvocate", "Evaluate everything from the end user's daily "
     "workflow: friction, cognitive load, failure recovery, and whether "
     "anyone will actually keep using this after week one."),
]

RACE_BRIEFS = [
    "Optimize strictly for implementation speed — the fastest correct path.",
    "Optimize strictly for robustness — assume hostile inputs and flaky deps.",
    "Optimize strictly for simplicity — fewest moving parts a stranger can maintain.",
    "Optimize strictly for performance — make it fast, then justify the cost.",
    "Optimize strictly for reversibility — easiest to undo if it's wrong.",
]

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position": {"type": "integer"},
                    "score": {"type": "number"},
                    "one_line": {"type": "string"},
                },
                "required": ["position", "score", "one_line"],
            },
        },
        "synthesis": {"type": "string"},
        "dissent": {"type": "array", "items": {"type": "string"}},
        "must_fix": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "recommended_next_step": {"type": "string"},
    },
    "required": ["scores", "synthesis", "dissent", "confidence"],
}


# ------------------------------------------------------------- delegation

def _extract_positions(raw_json: str, expected: int) -> List[Dict[str, Any]]:
    """Normalize delegate_task's batch result into position texts."""
    try:
        data = json.loads(raw_json)
    except Exception:
        return [{"text": raw_json[:3000], "status": "unknown", "label": f"member-{i+1}"}
                 for i in range(expected)]
    results = data.get("results") or data.get("tasks") or []
    positions: List[Dict[str, Any]] = []
    for i, entry in enumerate(results[:expected]):
        if not isinstance(entry, dict):
            positions.append({"text": str(entry)[:3000], "status": "unknown",
                              "label": f"member-{i+1}"})
            continue
        text = (entry.get("result") or entry.get("summary")
                or entry.get("output") or entry.get("error") or "")
        positions.append({
            "text": str(text)[:4000],
            "status": entry.get("status") or ("ok" if text else "empty"),
            "label": entry.get("goal", f"member-{i+1}")[:40],
        })
    while len(positions) < expected:
        positions.append({"text": "", "status": "missing",
                          "label": f"member-{len(positions)+1}"})
    return positions


def run_delegation(ctx, tasks: List[Dict[str, Any]], parent_agent=None) -> List[Dict[str, Any]]:
    kwargs: Dict[str, Any] = {}
    if parent_agent is not None:
        kwargs["parent_agent"] = parent_agent
    raw = ctx.dispatch_tool("delegate_task", {"tasks": tasks, "role": "leaf"}, **kwargs)
    return _extract_positions(raw, len(tasks))


# ----------------------------------------------------------------- judge

def _judge(ctx, question: str, positions: List[Dict[str, Any]],
           mode: str, judge_model: str = "") -> Dict[str, Any]:
    numbered = "\n\n".join(
        f"--- POSITION {i+1} ({p.get('label','?')}, status={p.get('status','?')}) ---\n"
        f"{p.get('text','') or '(no output)'}"
        for i, p in enumerate(positions))
    instructions = (
        "You are the impartial judge of an adversarial council. Read the "
        "question and every position. Score each position 0-10 on "
        "correctness, evidence, and usefulness. Then synthesize the best "
        "possible answer — DO NOT average positions; adopt the strongest "
        "arguments and say where they came from. Preserve real dissent: "
        "list every material disagreement that remains unresolved. "
        + ("This was a RED TEAM round: also produce must_fix — the concrete "
           "defects the architect must repair before the proposal ships. "
           if mode == "redteam" else "")
        + "End with a single recommended_next_step. confidence is 0-1."
    )
    try:
        kwargs: Dict[str, Any] = {}
        if judge_model:
            kwargs["model"] = judge_model  # trust-gated by the host
        result = ctx.llm.complete_structured(
            instructions=instructions,
            input=[f"QUESTION:\n{question}\n\n{numbered}"],
            json_schema=_VERDICT_SCHEMA,
            schema_name="council_verdict",
            purpose=f"council-judge-{mode}",
            **kwargs,
        )
        data = result.data if hasattr(result, "data") else None
        if isinstance(data, dict) and data.get("synthesis"):
            return data
    except Exception as exc:
        logger.debug("council: structured judge failed (%s), trying text mode", exc)

    # Fallback: plain completion + lenient JSON scrape.
    try:
        kwargs = {"model": judge_model} if judge_model else {}
        result = ctx.llm.complete(
            messages=[
                {"role": "system", "content": instructions +
                 " Respond with ONLY a JSON object matching this schema: "
                 + json.dumps(_VERDICT_SCHEMA)},
                {"role": "user", "content": f"QUESTION:\n{question}\n\n{numbered}"},
            ],
            purpose=f"council-judge-{mode}-text",
            **kwargs,
        )
        text = getattr(result, "text", "") or ""
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        return json.loads(text)
    except Exception as exc:
        logger.debug("council: text judge failed: %s", exc)
        return {
            "scores": [], "synthesis": "",
            "dissent": ["Judge unavailable — positions returned unsynthesized."],
            "must_fix": [], "confidence": 0.0,
            "recommended_next_step": "Read the raw positions and decide manually.",
        }


# ------------------------------------------------------------- orchestration

def convene(ctx, question: str, context: str = "", mode: str = "deliberate",
            councilors: int = 3, parent_agent=None, judge_model: str = "") -> Dict[str, Any]:
    councilors = max(2, min(5, int(councilors or 3)))
    mode = mode if mode in ("deliberate", "redteam", "race") else "deliberate"
    started = time.time()
    ctx_block = f"\n\nShared context (use it, verify it with your tools):\n{context}" if context else ""

    if mode == "redteam":
        architect = run_delegation(ctx, [{
            "goal": (
                "You are the Architect. Produce a concrete, complete proposal "
                "for the following — specific enough that someone could "
                "implement it tomorrow. Question:\n" + question + ctx_block),
            "context": context, "role": "leaf",
        }], parent_agent)
        proposal = architect[0]["text"] if architect else ""
        attack_tasks = []
        for name, brief in LENSES[1:councilors]:
            attack_tasks.append({
                "goal": (
                    f"You are the {name} on a red team. {brief}\n\nAttack "
                    f"this proposal — find concrete defects, not vibes. Cite "
                    f"the exact part you break and how.\n\nPROPOSAL:\n{proposal}"
                    f"\n\nORIGINAL QUESTION:\n{question}"),
                "context": "", "role": "leaf",
            })
        attackers = run_delegation(ctx, attack_tasks, parent_agent) if attack_tasks else []
        positions = [{"text": proposal, "status": "ok", "label": "Architect"}] + attackers
        labels = ["Architect"] + [n for n, _ in LENSES[1:councilors]][:len(attackers)]
        for i, p in enumerate(positions):
            p["label"] = labels[i] if i < len(labels) else f"attacker-{i}"
    elif mode == "race":
        tasks = [{
            "goal": (
                f"{brief}\n\nSolve this, end to end. You may use your tools "
                f"to inspect the environment first.\n\nTASK:\n{question}{ctx_block}"),
            "context": context, "role": "leaf",
        } for brief in RACE_BRIEFS[:councilors]]
        positions = run_delegation(ctx, tasks, parent_agent)
        for i, p in enumerate(positions):
            p["label"] = RACE_BRIEFS[i].split("—")[0].strip() if i < len(RACE_BRIEFS) else f"racer-{i+1}"
    else:  # deliberate
        tasks = [{
            "goal": (
                f"You are the {name} on a decision council. {brief}\n\n"
                f"Take a position on the question below. Use your tools to "
                f"check facts (read files, run commands, search) before "
                f"opining — positions grounded in evidence score higher. "
                f"~400 words max.\n\nQUESTION:\n{question}{ctx_block}"),
            "context": context, "role": "leaf",
        } for name, brief in LENSES[:councilors]]
        positions = run_delegation(ctx, tasks, parent_agent)
        for i, p in enumerate(positions):
            p["label"] = LENSES[i][0] if i < len(LENSES) else f"member-{i+1}"

    verdict = _judge(ctx, question, positions, mode, judge_model=judge_model)

    # attach scores to positions
    score_map = {}
    for s in verdict.get("scores") or []:
        try:
            score_map[int(s.get("position"))] = s
        except Exception:
            continue
    out_positions = []
    for i, p in enumerate(positions, start=1):
        s = score_map.get(i) or {}
        out_positions.append({
            "lens": p["label"],
            "status": p["status"],
            "score": s.get("score"),
            "judge_one_liner": s.get("one_line", ""),
            "position": p["text"],
        })

    return {
        "success": True,
        "mode": mode,
        "question": question,
        "council_size": len(positions),
        "duration_s": round(time.time() - started, 1),
        "synthesis": verdict.get("synthesis", ""),
        "confidence": verdict.get("confidence"),
        "positions": out_positions,
        "dissent": verdict.get("dissent") or [],
        "must_fix": verdict.get("must_fix") or [],
        "recommended_next_step": verdict.get("recommended_next_step", ""),
    }


def format_markdown(result: Dict[str, Any]) -> str:
    """Human-readable rendering for the /council slash command."""
    lines = [f"## Council verdict ({result['mode']}, {result['council_size']} members, "
             f"{result['duration_s']}s)\n"]
    conf = result.get("confidence")
    lines.append(f"**Synthesis** (confidence {conf if conf is not None else '—'}):\n"
                 f"{result.get('synthesis','') or '_(none) — judge unavailable_'}\n")
    if result.get("recommended_next_step"):
        lines.append(f"**Next step:** {result['recommended_next_step']}\n")
    lines.append("**Positions:**")
    for p in result["positions"]:
        score = f"{p['score']}/10" if p.get("score") is not None else "unscored"
        one = f" — {p['judge_one_liner']}" if p.get("judge_one_liner") else ""
        lines.append(f"- **{p['lens']}** ({p['status']}, {score}){one}")
    if result.get("dissent"):
        lines.append("\n**Unresolved dissent:**")
        lines.extend(f"- {d}" for d in result["dissent"])
    if result.get("must_fix"):
        lines.append("\n**Must fix before shipping:**")
        lines.extend(f"- {m}" for m in result["must_fix"])
    return "\n".join(lines)
