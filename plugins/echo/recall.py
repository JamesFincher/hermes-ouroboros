"""Echo recall engine — local-first proactive memory of past sessions.

Hermes already ships ``session_search`` (an agent-invoked FTS5 tool) and
pluggable memory providers (Honcho, Mem0, … — external services that
store *facts about you*).  Echo fills the gap between them: it needs no
external service and no model cooperation.  A ``pre_llm_call`` hook runs
an FTS5 query over the local session database every turn and injects the
most relevant past-session excerpts as context, automatically.

Ranking = BM25 (from SQLite FTS5) x recency boost x source weight:
  - cron sessions are demoted (repetitive vocabulary floods BM25)
  - subagent/tool sessions are excluded entirely
  - the current session is excluded
Sessions already injected for the current conversation are remembered,
so the same recall is never injected twice unless the query drifts.

Everything is read-only against state.db and every failure degrades to
"no context injected" — Echo can never break a turn.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------- config

_DEFAULTS = {
    "enabled": True,
    "max_hits": 3,            # past sessions surfaced per turn
    "char_budget": 1500,      # total injection budget
    "min_message_chars": 24,  # ignore "ok", "thanks", slash commands
    "recency_boost_days": 30, # sessions younger than this rank higher
    "deja_vu_days": 14,       # "you've been here before" threshold
}

_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while","for",
    "to","of","in","on","at","by","with","from","as","is","are","was","were",
    "be","been","being","do","does","did","have","has","had","i","you","he",
    "she","it","we","they","me","him","her","us","them","my","your","his",
    "its","our","their","this","that","these","those","there","here","what",
    "which","who","whom","how","why","can","could","should","would","will",
    "shall","may","might","must","not","no","yes","so","too","very","just",
    "about","into","over","under","again","once","also","please","make",
    "get","got","use","using","used","want","need","like","know","think",
    "see","look","take","give","go","going","went","come","say","said",
    "tell","told","ask","asked","help","let","put","try","trying","tried",
    "thing","things","something","anything","everything","way","really",
    "much","many","some","any","all","each","more","most","other","than",
    "now","new","one","two","out","up","down","only","own","same","still",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.#-]{2,}")


def extract_query_terms(text: str, max_terms: int = 8) -> List[str]:
    """Pull content terms out of a user message, preserving order by salience.

    Capitalized tokens (probable proper nouns: Redis, Django, HermesAgent)
    and tokens with digits/symbols (gpt-5, v0.14.0, c++) rank first.
    """
    words = _WORD_RE.findall(text or "")
    scored: List[tuple] = []
    seen: Set[str] = set()
    for i, w in enumerate(words):
        low = w.lower()
        if low in _STOPWORDS or low in seen or len(low) < 3:
            continue
        seen.add(low)
        salience = 0
        if w[0].isupper():
            salience += 2
        if any(c.isdigit() for c in w) or any(c in w for c in "+.#-_"):
            salience += 1
        if len(w) >= 8:
            salience += 1
        scored.append((-salience, i, w))
    scored.sort()
    return [w for _, _, w in scored[:max_terms]]


def build_fts_query(terms: List[str]) -> str:
    quoted = ['"' + t.replace('"', '""') + '"' for t in terms]
    return " OR ".join(quoted)


class RecallEngine:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = dict(_DEFAULTS)
        self._load_config()
        # per-conversation anti-spam: {session_id: {"terms": set, "sessions": set}}
        self._injected: Dict[str, Dict[str, Set]] = {}
        self.last_block: str = ""

    # ------------------------------------------------------------- config

    def _load_config(self) -> None:
        try:
            if self.config_path.exists():
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.config.update({k: v for k, v in data.items() if k in _DEFAULTS})
        except Exception as exc:
            logger.debug("echo: config load failed: %s", exc)

    def save_config(self) -> None:
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(
                json.dumps(self.config, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("echo: config save failed: %s", exc)

    def reset_session(self, session_id: str) -> None:
        self._injected.pop(session_id, None)

    # ------------------------------------------------------------- search

    def _open_db(self):
        from hermes_state import SessionDB  # local import: plugin loads in-process

        return SessionDB(read_only=True)

    def recall(self, query_text: str, *, exclude_session: str = "",
               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search past sessions. Returns ranked, deduped session hits."""
        terms = extract_query_terms(query_text)
        if not terms:
            return []
        limit = limit or int(self.config["max_hits"])
        try:
            db = self._open_db()
        except Exception as exc:
            logger.debug("echo: cannot open session db: %s", exc)
            return []
        try:
            rows = db.search_messages(
                build_fts_query(terms),
                exclude_sources=["subagent", "tool"],
                limit=40,
            )
        except Exception as exc:
            logger.debug("echo: search failed: %s", exc)
            return []
        finally:
            try:
                db.close()
            except Exception:
                pass

        now = time.time()
        boost_window = float(self.config["recency_boost_days"]) * 86400
        best_by_session: Dict[str, Dict[str, Any]] = {}
        for rank, row in enumerate(rows):
            sid = row.get("session_id") or ""
            if not sid or sid == exclude_session:
                continue
            score = 1.0 / (1.0 + rank)  # BM25 order → positional score
            started = row.get("session_started") or row.get("timestamp") or 0
            try:
                age = now - float(started)
                if age < boost_window:
                    score *= 1.0 + 0.5 * (1.0 - age / boost_window)
            except Exception:
                pass
            if row.get("source") == "cron":
                score *= 0.5
            prev = best_by_session.get(sid)
            if prev is None or score > prev["score"]:
                best_by_session[sid] = {"score": score, "row": row}

        hits = sorted(best_by_session.values(),
                      key=lambda h: h["score"], reverse=True)[:limit]
        return [
            {
                "session_id": sid,
                "score": round(h["score"], 4),
                "snippet": self._clean_snippet(h["row"].get("snippet") or ""),
                "role": h["row"].get("role") or "",
                "source": h["row"].get("source") or "",
                "session_started": h["row"].get("session_started"),
                "context": h["row"].get("context") or [],
                "terms": terms,
            }
            for sid, h in ((h["row"].get("session_id"), h) for h in hits)
        ]

    @staticmethod
    def _clean_snippet(snippet: str) -> str:
        return re.sub(r"\s+", " ", snippet.replace(">>>", "").replace("<<<", "")
                      .replace("...", "…")).strip()

    # --------------------------------------------------------- turn hook

    def build_turn_context(self, user_message: str, session_id: str) -> str:
        """The pre_llm_call payload. Returns '' when there's nothing to add."""
        if not self.config.get("enabled", True):
            return ""
        msg = (user_message or "").strip()
        if len(msg) < int(self.config["min_message_chars"]) or msg.startswith("/"):
            return ""

        terms = set(t.lower() for t in extract_query_terms(msg))
        if not terms:
            return ""
        state = self._injected.setdefault(session_id, {"terms": set(), "sessions": set()})
        # Query drift guard: if this turn's terms mostly overlap the last
        # injection's terms, the same recall is already in context.
        if state["terms"]:
            overlap = len(terms & state["terms"]) / max(1, len(terms | state["terms"]))
            if overlap > 0.7:
                return ""

        hits = self.recall(msg, exclude_session=session_id)
        fresh = [h for h in hits if h["session_id"] not in state["sessions"]]
        if not fresh:
            return ""

        budget = int(self.config["char_budget"])
        lines: List[str] = []
        deja_vu = False
        now = time.time()
        for h in fresh:
            snippet = h["snippet"][:320]
            if not snippet:
                continue
            when = ""
            try:
                ts = float(h["session_started"] or 0)
                when = time.strftime("%b %d", time.localtime(ts))
                if now - ts < float(self.config["deja_vu_days"]) * 86400:
                    deja_vu = True
            except Exception:
                pass
            ctx_roles = [c.get("role", "?") for c in h["context"]]
            lines.append(
                f"- [{when or 'earlier'} · {h['source'] or '?'} · session "
                f"{h['session_id'][:8]}] ({h['role']}) {snippet}"
                + (f"  (nearby: {' → '.join(ctx_roles)})" if ctx_roles else ""))
        if not lines:
            return ""

        block_lines = []
        if deja_vu:
            block_lines.append(
                "Note: the user appears to be returning to a topic discussed "
                "recently — prefer continuity over starting from scratch.")
        block_lines.extend(lines)
        block = "\n".join(block_lines)[:budget]
        block = (
            "<echo_recall source=\"local session history, auto-injected\">\n"
            f"{block}\n"
            "Mention this only if relevant; use session_search with these "
            "session ids to dig deeper.\n</echo_recall>"
        )
        state["terms"] |= terms
        state["sessions"] |= {h["session_id"] for h in fresh}
        self.last_block = block
        return block
