"""Archive context engine — lossless compression for Hermes.

The built-in ContextCompressor is *summarize-and-discard*: the middle of
the conversation is folded into a summary and the originals are gone from
the model's reach (they persist in the session DB, but the agent has no
live way to search them — session_search is a separate, manual tool the
model must remember to call).

ArchiveEngine wraps the built-in compressor and changes the contract to
*seal-then-summarize*:

  1. should_compress / update_from_response / compress policy are all
     delegated to a real ContextCompressor instance, so trigger behavior,
     thresholds and summary quality are identical to stock Hermes.
  2. Before each compress(), the window about to leave context is sealed
     into a local FTS5 archive (~/.hermes/archive.db), deduped per message.
  3. At real session end, the remaining tail is sealed too — nothing that
     was ever in context is unrecoverable.
  4. The engine exposes archive_search / archive_expand / archive_status
     tools through get_tool_schemas(), so the *agent itself* can pull
     sealed context back mid-conversation.  This is the LCM-shaped
     capability the ContextEngine ABC was designed for (see its docstring
     references to lcm_grep / lcm_expand) but which no bundled engine ships.

Host integration notes (verified against agent/agent_init.py):
  - The plugin registers one engine singleton; the host deep-copies it per
    agent.  __deepcopy__ therefore copies only scalar config — the inner
    compressor is rebuilt in update_model() and the store re-opens lazily.
  - update_model() is called right after selection with the active model's
    context length and credentials; that is where the inner compressor is
    constructed.
  - bind_session_state(session_db=..., session_id=...) is called when the
    host has a session DB handy; we use it to learn our session id early.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)


class ArchiveEngine(ContextEngine):
    """ContextEngine that archives before it compresses."""

    def __init__(self, threshold_percent: float = 0.50,
                 protect_first_n: int = 3, protect_last_n: int = 20,
                 archive_dir: str = ""):
        self._cfg = {
            "threshold_percent": threshold_percent,
            "protect_first_n": protect_first_n,
            "protect_last_n": protect_last_n,
            "archive_dir": archive_dir,
        }
        self._inner = None           # ContextCompressor, built in update_model
        self._store = None           # ArchiveStore, opened lazily
        self._session_id = ""
        self._seen_hashes: set = set()   # per-process message dedupe
        self._sealed_chunks: List[int] = []

    # -- identity ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "archive"

    # -- host lifecycle ---------------------------------------------------

    def __deepcopy__(self, memo):
        # Only scalar config crosses the copy boundary; stateful members
        # (inner compressor, store connection, dedupe set) are rebuilt.
        return type(self)(**self._cfg)

    def _ensure_store(self):
        if self._store is None:
            from pathlib import Path

            from hermes_constants import get_hermes_home

            from .store import ArchiveStore

            root = self._cfg["archive_dir"] or (Path(get_hermes_home()))
            self._store = ArchiveStore(Path(root) / "archive.db")
        return self._store

    def update_model(self, model: str, context_length: int, base_url: str = "",
                     api_key: str = "", provider: str = "", api_mode: str = "") -> None:
        from agent.context_compressor import ContextCompressor

        self._inner = ContextCompressor(
            model=model,
            threshold_percent=self._cfg["threshold_percent"],
            protect_first_n=self._cfg["protect_first_n"],
            protect_last_n=self._cfg["protect_last_n"],
            quiet_mode=True,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            api_mode=api_mode,
        )
        # Mirror token state for run_agent.py's status readers.
        self.context_length = self._inner.context_length or context_length
        self.threshold_tokens = self._inner.threshold_tokens
        self.threshold_percent = self._inner.threshold_percent
        self.protect_first_n = self._inner.protect_first_n
        self.protect_last_n = self._inner.protect_last_n

    def bind_session_state(self, session_db=None, session_id: str = "") -> None:
        if session_id:
            self._session_id = session_id

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or self._session_id
        self._seen_hashes.clear()
        self._sealed_chunks.clear()
        if self._inner is not None and hasattr(self._inner, "on_session_start"):
            try:
                self._inner.on_session_start(session_id, **kwargs)
            except Exception:
                pass

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        # Final flush: seal whatever is still live (incl. the tail the
        # compressor always keeps) so the whole session is recoverable.
        try:
            self._seal(messages or [], reason="final")
        except Exception as exc:
            logger.debug("archive: final seal failed: %s", exc)
        if self._inner is not None and hasattr(self._inner, "on_session_end"):
            try:
                self._inner.on_session_end(session_id, messages)
            except Exception:
                pass

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._seen_hashes.clear()
        self._sealed_chunks.clear()
        if self._inner is not None and hasattr(self._inner, "on_session_reset"):
            try:
                self._inner.on_session_reset()
            except Exception:
                pass

    # -- token bookkeeping (delegated) -------------------------------------

    def _mirror(self) -> None:
        inner = self._inner
        if inner is None:
            return
        self.last_prompt_tokens = inner.last_prompt_tokens
        self.last_completion_tokens = inner.last_completion_tokens
        self.last_total_tokens = inner.last_total_tokens
        self.compression_count = inner.compression_count

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        if self._inner is not None:
            self._inner.update_from_response(usage)
        self._mirror()

    def should_compress(self, prompt_tokens: int = None) -> bool:
        if self._inner is None:
            return False
        return self._inner.should_compress(prompt_tokens)

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        if self._inner is None:
            return False
        return self._inner.should_compress_preflight(messages)

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        if self._inner is None:
            return False
        return self._inner.should_defer_preflight_to_real_usage(rough_tokens)

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        if self._inner is None:
            return True
        return self._inner.has_content_to_compress(messages)

    # -- the money shot ------------------------------------------------------

    @staticmethod
    def _msg_fingerprint(m: Dict[str, Any]) -> str:
        content = m.get("content")
        if isinstance(content, list):
            content = json.dumps(content, default=str)[:2000]
        blob = f"{m.get('role')}|{m.get('tool_name') or ''}|{str(content)[:4000]}"
        return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()

    def _middle_window(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The slice compression is about to remove: after the protected head
        (system + first N non-system), before the protected tail."""
        head = 0
        seen_non_system = 0
        for i, m in enumerate(messages):
            if m.get("role") == "system":
                head = i + 1
                continue
            seen_non_system += 1
            if seen_non_system <= self.protect_first_n:
                head = i + 1
            else:
                break
        tail = max(0, len(messages) - self.protect_last_n)
        if tail <= head:
            return []
        return messages[head:tail]

    def _seal(self, messages: List[Dict[str, Any]], reason: str) -> Optional[int]:
        fresh = [m for m in messages
                 if self._msg_fingerprint(m) not in self._seen_hashes]
        if not fresh:
            return None
        for m in fresh:
            self._seen_hashes.add(self._msg_fingerprint(m))
        chunk_id = self._ensure_store().seal_chunk(
            self._session_id or "unknown", fresh, reason=reason)
        if chunk_id is not None:
            self._sealed_chunks.append(chunk_id)
        return chunk_id

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None,
                 focus_topic: str = None) -> List[Dict[str, Any]]:
        # 1. Seal the about-to-vanish window (never raises into the agent loop).
        try:
            self._seal(self._middle_window(messages),
                       reason="focus" if focus_topic else "auto")
        except Exception as exc:
            logger.debug("archive: seal before compress failed: %s", exc)
        # 2. Delegate the actual compression to stock Hermes logic.
        if self._inner is None:
            return messages
        out = self._inner.compress(messages, current_tokens=current_tokens,
                                   focus_topic=focus_topic)
        self._mirror()
        return out

    # -- engine tools (the LCM-shaped surface) --------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "archive_search",
                "description": (
                    "Search the Deep Archive: every piece of context that "
                    "compression removed from this and past sessions, sealed "
                    "verbatim and FTS5-indexed. Use when you vaguely remember "
                    "something from earlier (a decision, an error, a path, a "
                    "command) that is no longer in context. Returns matching "
                    "chunks with snippets; follow up with archive_expand."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "FTS5 query — keywords, "
                                                 "\"exact phrase\", OR/NOT."},
                        "limit": {"type": "integer",
                                  "description": "Max chunks (default 5)."},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "archive_expand",
                "description": (
                    "Rehydrate a sealed archive chunk: returns the original "
                    "messages from that compressed-away window, verbatim. "
                    "Use after archive_search identifies a relevant chunk."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "integer",
                                     "description": "Chunk id from archive_search."},
                    },
                    "required": ["chunk_id"],
                },
            },
            {
                "name": "archive_status",
                "description": "Deep Archive stats: sealed chunks, messages, "
                               "approximate tokens preserved, DB size.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        store = self._ensure_store()
        if name == "archive_search":
            hits = store.search(str((args or {}).get("query") or ""),
                                limit=int((args or {}).get("limit") or 5))
            return json.dumps({"success": True, "hits": hits,
                               "hint": "archive_expand(chunk_id) to read a chunk verbatim."},
                              ensure_ascii=False, default=str)
        if name == "archive_expand":
            return json.dumps(store.expand(int((args or {}).get("chunk_id") or 0)),
                              ensure_ascii=False, default=str)
        if name == "archive_status":
            st = store.status()
            st["session_chunks_this_agent"] = list(self._sealed_chunks)
            return json.dumps({"success": True, "archive": st}, default=str)
        return json.dumps({"error": f"Unknown archive tool: {name}"})

    # -- status ---------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        base = super().get_status()
        self._mirror()
        base.update({
            "last_prompt_tokens": self.last_prompt_tokens,
            "compression_count": self.compression_count,
            "engine": "archive",
            "sealed_chunks": len(self._sealed_chunks),
        })
        return base
