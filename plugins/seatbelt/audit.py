"""Seatbelt audit log — append-only SQLite record of every tool call.

Lives at ~/.hermes/seatbelt/audit.db.  Args are never stored verbatim:
only a SHA-256 digest and byte size, so the log proves *what shape* of
call happened without becoming a secret spill of its own.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT,
    tool_name TEXT NOT NULL,
    status TEXT,
    decision TEXT NOT NULL,
    rule TEXT,
    duration_ms REAL,
    args_digest TEXT,
    args_bytes INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_tool ON events(tool_name, ts);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, ts);
"""

_RETENTION_DAYS = 90


class AuditLog:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @staticmethod
    def _digest(args: Dict[str, Any]) -> tuple[str, int]:
        try:
            blob = json.dumps(args or {}, default=str, ensure_ascii=False).encode("utf-8", "replace")
        except Exception:
            blob = str(args).encode("utf-8", "replace")
        return hashlib.sha256(blob).hexdigest()[:16], len(blob)

    def record(
        self,
        *,
        tool_name: str,
        args: Optional[Dict[str, Any]],
        session_id: str = "",
        status: str = "",
        decision: str = "allow",
        rule: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        digest, nbytes = self._digest(args or {})
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO events (ts, session_id, tool_name, status, decision, rule,"
                    " duration_ms, args_digest, args_bytes) VALUES (?,?,?,?,?,?,?,?,?)",
                    (time.time(), session_id, tool_name, status, decision, rule,
                     float(duration_ms or 0.0), digest, nbytes),
                )
                self._conn.commit()
        except Exception as exc:
            logger.debug("seatbelt audit write failed: %s", exc)

    def tail(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT ts, session_id, tool_name, status, decision, rule, duration_ms"
                " FROM events ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def top_tools(self, days: int = 7, limit: int = 10) -> List[tuple]:
        since = time.time() - days * 86400
        with self._lock:
            cur = self._conn.execute(
                "SELECT tool_name, COUNT(*), ROUND(AVG(duration_ms),1),"
                " SUM(CASE WHEN decision != 'allow' THEN 1 ELSE 0 END)"
                " FROM events WHERE ts > ? GROUP BY tool_name ORDER BY 2 DESC LIMIT ?",
                (since, int(limit)),
            )
            return cur.fetchall()

    def counts(self) -> Dict[str, Any]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            flagged = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE decision != 'allow'"
            ).fetchone()[0]
        return {"total_events": total, "flagged_events": flagged,
                "db_path": str(self.db_path)}

    def prune(self, retention_days: int = _RETENTION_DAYS) -> int:
        cutoff = time.time() - retention_days * 86400
        with self._lock:
            cur = self._conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
            self._conn.commit()
            return cur.rowcount or 0
