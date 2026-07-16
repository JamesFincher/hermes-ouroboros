"""Ouroboros — the flywheel keystone.

Six plugins, one loop:

    REMEMBER   echo + archive  — nothing you did is ever lost
    TRUST      seatbelt        — so the agent may act without fear
    DECIDE     council         — hard questions get argued, not guessed
    AUTOMATE   autopilot       — repeated chores become cron jobs
    LEARN      forge           — your best sessions become training data
      └─ which makes tomorrow's sessions better — which feed REMEMBER.

This plugin is the dashboard for that loop.  It reads each sibling
plugin's on-disk state (never their code — siblings stay decoupled),
shows which flywheel stages are spinning, and surfaces the single most
valuable next action.

    /ouroboros        flywheel status + next action
    /ouroboros story  the legend, in one screen
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


def _home() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home())


def _enabled_plugins() -> set:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        plugins = cfg.get("plugins") or {}
        enabled = plugins.get("enabled")
        if enabled is None:  # absent = everything discovered is enabled
            return {"seatbelt", "echo", "archive", "council", "autopilot",
                    "forge", "ouroboros"}
        return {str(x) for x in (enabled or [])}
    except Exception:
        return set()


# ---------------------------------------------------------- stage probes
# Each probe reads ONLY sibling state files and returns (line, next_action).

def _probe_echo(home: Path) -> tuple[str, str | None]:
    cfg = home / "echo" / "config.json"
    enabled = True
    try:
        if cfg.exists():
            enabled = bool(json.loads(cfg.read_text()).get("enabled", True))
    except Exception:
        pass
    if not enabled:
        return "echo: installed, auto-recall OFF (/echo on to resume)", \
            "re-enable recall with /echo on"
    return "echo: auto-recall armed — past sessions surface as you type", None


def _probe_archive(home: Path) -> tuple[str, str | None]:
    db = home / "archive.db"
    if not db.exists():
        return "archive: no sealed context yet (needs context.engine: archive)", \
            "set context.engine: archive in config.yaml to stop forgetting"
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.0)
        chunks, tokens = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(approx_tokens),0) FROM chunks").fetchone()
        conn.close()
        return (f"archive: {chunks} context windows sealed "
                f"(~{int(tokens):,} tokens preserved)"), None
    except Exception:
        return "archive: store present but unreadable", None


def _probe_seatbelt(home: Path) -> tuple[str, str | None]:
    db = home / "seatbelt" / "audit.db"
    if not db.exists():
        return "seatbelt: watching (no calls audited yet)", None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.0)
        total, flagged = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN decision != 'allow' THEN 1 ELSE 0 END)"
            " FROM events").fetchone()
        conn.close()
        return f"seatbelt: {int(total or 0):,} calls audited, {int(flagged or 0)} flagged", None
    except Exception:
        return "seatbelt: audit unreadable", None


def _probe_council(home: Path) -> tuple[str, str | None]:
    return "council: ready — /council <hard question> when the stakes justify it", None


def _probe_autopilot(home: Path) -> tuple[str, str | None]:
    path = home / "autopilot" / "suggestions.json"
    if not path.exists():
        return "autopilot: learning your rhythms (no suggestions yet)", None
    try:
        items = json.loads(path.read_text())
        pending = sum(1 for s in items if s.get("status") == "pending")
        approved = sum(1 for s in items if s.get("status") == "approved")
        line = f"autopilot: {pending} automations awaiting you, {approved} already running"
        action = f"{pending} automation(s) ready — /autopilot list" if pending else None
        return line, action
    except Exception:
        return "autopilot: suggestions unreadable", None


def _probe_forge(home: Path) -> tuple[str, str | None]:
    idx = home / "forge" / "index.json"
    if not idx.exists():
        return "forge: dataset empty — /forge harvest to distill your history", \
            "run /forge harvest to start your personal dataset"
    try:
        data = json.loads(idx.read_text())
        return f"forge: {len(data)} sessions distilled into dataset.jsonl", None
    except Exception:
        return "forge: index unreadable", None


_STAGES = [
    ("REMEMBER", "echo", _probe_echo),
    ("PRESERVE", "archive", _probe_archive),
    ("TRUST", "seatbelt", _probe_seatbelt),
    ("DECIDE", "council", _probe_council),
    ("AUTOMATE", "autopilot", _probe_autopilot),
    ("LEARN", "forge", _probe_forge),
]

_STORY = """\
The snake that eats its tail.

Every session you run produces exhaust: tool calls, compressions,
repeated requests, hard decisions, finished tasks.  Stock agents let the
exhaust dissipate.  Ouroboros eats it.

  echo + archive  turn your history into context that comes back.
  seatbelt        turns that history into permission to act unsupervised.
  council         turns hard questions into argued verdicts.
  autopilot       turns repetition into scheduled automations.
  forge           turns everything else into training data.

Run it long enough and the loop is the point: better memory → better
sessions → better data → a better agent → better memory.
"""


def _cmd_ouroboros(raw_args: str) -> str:
    args = (raw_args or "").strip().lower()
    if args == "story":
        return _STORY

    home = _home()
    enabled = _enabled_plugins()
    lines = ["The Ouroboros flywheel — " + time.strftime("%Y-%m-%d %H:%M"), ""]
    actions: list[str] = []
    installed = 0
    for stage, plugin, probe in _STAGES:
        if plugin not in enabled:
            lines.append(f"  ○ {stage:<9} {plugin}: not enabled "
                         f"(hermes plugins enable {plugin})")
            actions.append(f"enable {plugin} to spin the {stage.lower()} stage")
            continue
        installed += 1
        line, action = probe(home)
        lines.append(f"  ● {stage:<9} {line}")
        if action:
            actions.append(action)
    lines.append("")
    if installed < len(_STAGES):
        lines.append(f"{installed}/{len(_STAGES)} stages spinning.")
    else:
        lines.append("All six stages spinning. The snake is fed.")
    if actions:
        lines.append("")
        lines.append("Next best action: " + actions[0])
    return "\n".join(lines)


def register(ctx):
    ctx.register_command(
        "ouroboros", _cmd_ouroboros,
        description="Flywheel status for the Ouroboros plugin pack",
        args_hint="[story]",
    )
