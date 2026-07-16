"""Deep Archive store — SQLite + FTS5 sidecar for compressed-away context.

Layout (at ~/.hermes/archive.db):

  chunks          one row per sealed context window (a compression event
                  or a session-final flush)
  chunk_messages  the individual messages inside each chunk
  archive_fts     FTS5 index over message content (external-content mode
                  is avoided on purpose: self-contained keeps deletes and
                  rebuilds trivial)

The store is process-local (opened lazily, guarded by a lock) and safe
to open from several Hermes processes at once (WAL mode).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    reason TEXT NOT NULL,
    created_at REAL NOT NULL,
    message_count INTEGER NOT NULL,
    approx_tokens INTEGER NOT NULL,
    head_preview TEXT,
    tail_preview TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id, seq);
CREATE TABLE IF NOT EXISTS chunk_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id),
    idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    tool_name TEXT,
    content TEXT,
    ts REAL
);
CREATE INDEX IF NOT EXISTS idx_chunk_messages_chunk ON chunk_messages(chunk_id, idx);
CREATE VIRTUAL TABLE IF NOT EXISTS archive_fts USING fts5(
    content,
    tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS archive_fts_ai AFTER INSERT ON chunk_messages BEGIN
    INSERT INTO archive_fts(rowid, content) VALUES (new.id, COALESCE(new.content,''));
END;
CREATE TRIGGER IF NOT EXISTS archive_fts_ad AFTER DELETE ON chunk_messages BEGIN
    DELETE FROM archive_fts WHERE rowid = old.id;
END;
"""

# Mirror of the host's stopword-agnostic preview shape.
def _preview(text: str, n: int = 160) -> str:
    text = " ".join((text or "").split())
    return text[:n]


class ArchiveStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------- write

    def seal_chunk(self, session_id: str, messages: List[Dict[str, Any]],
                   reason: str = "auto") -> Optional[int]:
        """Persist a window of messages as a new chunk. Returns chunk id."""
        rows = []
        approx = 0
        for i, m in enumerate(messages):
            role = str(m.get("role") or "?")
            if role == "system":
                continue  # the system prompt is reconstructed, not archived
            content = m.get("content")
            if isinstance(content, list):  # multimodal → text-only summary
                parts = [p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(t for t in parts if t)
            content = str(content or "")
            tool_name = str(m.get("tool_name") or "")
            tool_calls = m.get("tool_calls")
            if tool_calls and not content:
                try:
                    content = "[tool calls] " + json.dumps(tool_calls, default=str)[:800]
                except Exception:
                    content = "[tool calls]"
            if not content and not tool_name:
                continue
            approx += max(1, len(content) // 4)
            rows.append((i, role, tool_name or None, content,
                         float(m.get("timestamp") or time.time())))
        if not rows:
            return None

        with self._lock:
            seq = (self._conn.execute(
                "SELECT COALESCE(MAX(seq),0)+1 FROM chunks WHERE session_id=?",
                (session_id,)).fetchone()[0])
            cur = self._conn.execute(
                "INSERT INTO chunks (session_id, seq, reason, created_at,"
                " message_count, approx_tokens, head_preview, tail_preview)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (session_id, seq, reason, time.time(), len(rows), approx,
                 _preview(rows[0][3]), _preview(rows[-1][3])))
            chunk_id = cur.lastrowid
            self._conn.executemany(
                "INSERT INTO chunk_messages (chunk_id, idx, role, tool_name,"
                " content, ts) VALUES (?,?,?,?,?,?)",
                [(chunk_id, idx, role, tool, content, ts)
                 for idx, role, tool, content, ts in rows])
            self._conn.commit()
        logger.info("archive: sealed chunk %s (%d messages, ~%d tokens, %s)",
                    chunk_id, len(rows), approx, reason)
        return chunk_id

    # -------------------------------------------------------------- read

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """FTS5 search → chunks ranked by best message hit, with snippets."""
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT cm.chunk_id AS chunk_id,
                       snippet(archive_fts, 0, '>>>', '<<<', '…', 24) AS snip,
                       cm.role AS role, cm.tool_name AS tool_name, cm.ts AS ts,
                       rank
                FROM archive_fts
                JOIN chunk_messages cm ON cm.id = archive_fts.rowid
                WHERE archive_fts MATCH ?
                ORDER BY rank
                LIMIT 60
                """, (query,))
            rows = cur.fetchall()
        best: Dict[int, Dict[str, Any]] = {}
        for r in rows:
            cid = r["chunk_id"]
            if cid not in best:
                best[cid] = {"chunk_id": cid, "snippets": [], "rank": r["rank"]}
            if len(best[cid]["snippets"]) < 3:
                snip = (r["snip"] or "").replace(">>>", "").replace("<<<", "")
                best[cid]["snippets"].append(
                    {"role": r["role"], "tool": r["tool_name"], "text": snip})
        if not best:
            return []
        with self._lock:
            out = []
            for cid, hit in sorted(best.items(), key=lambda kv: kv[1]["rank"]):
                meta = self._conn.execute(
                    "SELECT session_id, seq, reason, created_at, message_count,"
                    " approx_tokens FROM chunks WHERE id=?", (cid,)).fetchone()
                if not meta:
                    continue
                out.append({
                    "chunk_id": cid,
                    "session_id": meta["session_id"],
                    "seq": meta["seq"],
                    "reason": meta["reason"],
                    "created_at": meta["created_at"],
                    "message_count": meta["message_count"],
                    "approx_tokens": meta["approx_tokens"],
                    "snippets": hit["snippets"],
                })
                if len(out) >= limit:
                    break
        return out

    def expand(self, chunk_id: int, max_chars: int = 6000) -> Dict[str, Any]:
        with self._lock:
            meta = self._conn.execute(
                "SELECT * FROM chunks WHERE id=?", (int(chunk_id),)).fetchone()
            if not meta:
                return {"error": f"chunk {chunk_id} not found"}
            cur = self._conn.execute(
                "SELECT idx, role, tool_name, content, ts FROM chunk_messages"
                " WHERE chunk_id=? ORDER BY idx", (int(chunk_id),))
            messages = []
            total = 0
            truncated = False
            for r in cur.fetchall():
                line = f"[{r['role']}{'/' + r['tool_name'] if r['tool_name'] else ''}] {r['content']}"
                if total + len(line) > max_chars:
                    truncated = True
                    break
                messages.append({"idx": r["idx"], "role": r["role"],
                                 "tool_name": r["tool_name"],
                                 "content": r["content"], "ts": r["ts"]})
                total += len(line)
        return {
            "chunk_id": int(chunk_id),
            "session_id": meta["session_id"],
            "reason": meta["reason"],
            "created_at": meta["created_at"],
            "messages": messages,
            "truncated": truncated,
            "message_count": meta["message_count"],
        }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            msgs = self._conn.execute("SELECT COUNT(*) FROM chunk_messages").fetchone()[0]
            tokens = self._conn.execute(
                "SELECT COALESCE(SUM(approx_tokens),0) FROM chunks").fetchone()[0]
            last = self._conn.execute(
                "SELECT MAX(created_at) FROM chunks").fetchone()[0]
            sessions = self._conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM chunks").fetchone()[0]
        size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {"chunks": chunks, "messages": msgs, "approx_tokens": tokens,
                "sessions": sessions, "last_sealed": last,
                "db_bytes": size, "db_path": str(self.db_path)}
