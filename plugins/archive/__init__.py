"""Archive — lossless context engine for Hermes.

Registers the ``archive`` context engine.  Two-step activation (this is
how Hermes gates engine selection — the plugin must be enabled *and* the
engine must be chosen):

    hermes plugins enable archive
    # then in ~/.hermes/config.yaml:
    #   context:
    #     engine: archive

The engine wraps the built-in compressor: identical trigger and summary
behavior, but every message window that leaves the context window is
sealed into ~/.hermes/archive.db (FTS5) first, and the agent gains
archive_search / archive_expand / archive_status tools to rehydrate it.

The /archive command queries the archive directly (read-only), so it
works no matter which agent process sealed the chunks:
    /archive status
    /archive search <query>
    /archive expand <chunk_id>
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _open_store_ro():
    from hermes_constants import get_hermes_home

    from .store import ArchiveStore

    return ArchiveStore(Path(get_hermes_home()) / "archive.db")


def _cmd_archive(raw_args: str) -> str:
    parts = (raw_args or "").strip().split(maxsplit=1)
    sub = parts[0].lower() if parts and parts[0] else "status"
    try:
        store = _open_store_ro()
    except Exception as exc:
        return f"Archive unavailable: {exc}"

    if sub == "status":
        st = store.status()
        last = (time.strftime("%Y-%m-%d %H:%M", time.localtime(st["last_sealed"]))
                if st["last_sealed"] else "never")
        return (
            "Deep Archive\n"
            f"  Chunks sealed : {st['chunks']} across {st['sessions']} sessions\n"
            f"  Messages kept : {st['messages']} (~{st['approx_tokens']:,} tokens)\n"
            f"  Last sealed   : {last}\n"
            f"  Store         : {st['db_path']} ({st['db_bytes'] / 1e6:.1f} MB)\n"
            "Engine active only if context.engine: archive is set in config.yaml."
        )

    if sub == "search" and len(parts) > 1:
        hits = store.search(parts[1], limit=5)
        if not hits:
            return "No archive chunks matched."
        out = []
        for h in hits:
            when = time.strftime("%b %d %H:%M", time.localtime(h["created_at"]))
            snips = " / ".join(s["text"] for s in h["snippets"][:2])
            out.append(f"chunk {h['chunk_id']} [{when} · {h['reason']} · "
                       f"{h['message_count']} msgs] {snips}")
        return "Archive hits:\n" + "\n".join(out) + \
            "\n\n/archive expand <chunk_id> to read one verbatim."

    if sub == "expand" and len(parts) > 1:
        try:
            cid = int(parts[1].strip())
        except ValueError:
            return "Usage: /archive expand <chunk_id>"
        data = store.expand(cid, max_chars=3000)
        if "error" in data:
            return data["error"]
        lines = [f"Chunk {cid} — {data['message_count']} messages "
                 f"({data['reason']}, session {data['session_id'][:8]}…):"]
        for m in data["messages"][:20]:
            content = " ".join(str(m["content"]).split())[:200]
            lines.append(f"  [{m['role']}] {content}")
        if data.get("truncated"):
            lines.append("  … (truncated — use the archive_expand tool for more)")
        return "\n".join(lines)

    return "Usage: /archive [status|search <query>|expand <chunk_id>]"


def register(ctx):
    from .engine import ArchiveEngine

    ctx.register_context_engine(ArchiveEngine())
    ctx.register_command(
        "archive", _cmd_archive,
        description="Deep Archive: status, search, expand sealed context",
        args_hint="[status|search <query>|expand <chunk_id>]",
    )
