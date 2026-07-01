"""SQLite persistence for cross-session student progress.

Phase 4 JSON profiles remain the in-session working copy; this module is
the durable store so progress survives between days. All schema creation
is idempotent — call :func:`init_db` on every startup.

The ``students.needs_review`` column stores a JSON array of concept ids
(a small extension beyond the base four-table schema so review flags
round-trip with :class:`~socratix.student_model.StudentProfile`).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from socratix.student_model import (
    StudentProfile,
    diagnostic_to_graph_status,
    utc_now_iso,
)

DEFAULT_DB_PATH: Path = Path(__file__).resolve().parent.parent / "socratix.sqlite"

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS students (
        id           TEXT PRIMARY KEY,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL,
        needs_review TEXT NOT NULL DEFAULT '[]'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id     TEXT NOT NULL,
        target_concept TEXT,
        started_at     TEXT NOT NULL,
        ended_at       TEXT,
        FOREIGN KEY (student_id) REFERENCES students(id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS concept_status (
        student_id            TEXT NOT NULL,
        concept_id            TEXT NOT NULL,
        status                TEXT NOT NULL,
        diagnostic_label      TEXT,
        confidence            REAL NOT NULL,
        misconception_summary TEXT,
        rationale             TEXT,
        last_seen             TEXT NOT NULL,
        PRIMARY KEY (student_id, concept_id),
        FOREIGN KEY (student_id) REFERENCES students(id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  INTEGER NOT NULL,
        concept_id  TEXT,
        role        TEXT NOT NULL,
        kind        TEXT NOT NULL,
        content     TEXT NOT NULL,
        payload     TEXT,
        timestamp   TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_status_student ON concept_status(student_id);",
)


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create tables if missing and return a connection with ``Row`` factory.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        An open :class:`sqlite3.Connection`. Caller is responsible for
        closing it (or holding it in ``st.session_state`` for Streamlit).

    Note:
        ``check_same_thread=False`` is required for Streamlit, which may
        invoke the script from different worker threads across reruns.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)
    conn.commit()
    return conn


def upsert_student(
    conn: sqlite3.Connection,
    student_id: str,
    *,
    created_at: str | None = None,
    updated_at: str | None = None,
    needs_review: list[str] | None = None,
) -> None:
    """Insert or update a student row.

    Args:
        conn: Open database connection.
        student_id: Primary key for the student.
        created_at: Set on insert; preserved on update when omitted.
        updated_at: Always refreshed when provided.
        needs_review: JSON-serializable list of concept ids needing review.
    """
    now = utc_now_iso()
    created = created_at or now
    updated = updated_at or now
    review_json = json.dumps(needs_review or [])

    conn.execute(
        """
        INSERT INTO students (id, created_at, updated_at, needs_review)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            updated_at = excluded.updated_at,
            needs_review = excluded.needs_review
        """,
        (student_id, created, updated, review_json),
    )
    conn.commit()


def start_session(
    conn: sqlite3.Connection,
    student_id: str,
    target_concept: str | None,
) -> int:
    """Begin a new learning session and return its id.

    Args:
        conn: Open database connection.
        student_id: Student owning the session.
        target_concept: Optional target concept id for this session.

    Returns:
        The new session's integer primary key.
    """
    upsert_student(conn, student_id)
    cursor = conn.execute(
        """
        INSERT INTO sessions (student_id, target_concept, started_at)
        VALUES (?, ?, ?)
        """,
        (student_id, target_concept, utc_now_iso()),
    )
    conn.commit()
    return int(cursor.lastrowid)


def end_session(conn: sqlite3.Connection, session_id: int) -> None:
    """Mark a session as ended.

    Args:
        conn: Open database connection.
        session_id: Session to close.
    """
    conn.execute(
        "UPDATE sessions SET ended_at = ? WHERE id = ?",
        (utc_now_iso(), session_id),
    )
    conn.commit()


def save_concept_status(
    conn: sqlite3.Connection,
    student_id: str,
    concept_id: str,
    status: str,
    diagnostic_label: str | None,
    confidence: float,
    misconception_summary: str | None,
    rationale: str | None,
    *,
    last_seen: str | None = None,
) -> None:
    """Upsert one concept's status for a student.

    Args:
        conn: Open database connection.
        student_id: Student primary key.
        concept_id: Concept graph node id.
        status: Graph-level label (``known``, ``unknown``, etc.).
        diagnostic_label: Diagnostic label (``confirmed_known``, etc.).
        confidence: Model confidence in ``[0, 1]``.
        misconception_summary: Optional wrong-belief summary.
        rationale: One-sentence rationale.
        last_seen: ISO timestamp; defaults to now.
    """
    conn.execute(
        """
        INSERT INTO concept_status (
            student_id, concept_id, status, diagnostic_label,
            confidence, misconception_summary, rationale, last_seen
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(student_id, concept_id) DO UPDATE SET
            status = excluded.status,
            diagnostic_label = excluded.diagnostic_label,
            confidence = excluded.confidence,
            misconception_summary = excluded.misconception_summary,
            rationale = excluded.rationale,
            last_seen = excluded.last_seen
        """,
        (
            student_id,
            concept_id,
            status,
            diagnostic_label,
            float(confidence),
            misconception_summary,
            rationale,
            last_seen or utc_now_iso(),
        ),
    )
    conn.commit()


def load_student_state(conn: sqlite3.Connection, student_id: str) -> dict[str, dict[str, Any]]:
    """Load all persisted concept statuses for a student.

    Args:
        conn: Open database connection.
        student_id: Student primary key.

    Returns:
        ``{concept_id: {status, diagnostic_label, confidence, ...}}`` where
        ``status`` is the graph-level label and ``diagnostic_label`` is the
        verbose diagnostic label when stored.
    """
    rows = conn.execute(
        """
        SELECT concept_id, status, diagnostic_label, confidence,
               misconception_summary, rationale, last_seen
        FROM concept_status
        WHERE student_id = ?
        ORDER BY concept_id
        """,
        (student_id,),
    ).fetchall()

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        result[row["concept_id"]] = {
            "status": row["status"],
            "diagnostic_label": row["diagnostic_label"],
            "confidence": row["confidence"],
            "misconception_summary": row["misconception_summary"],
            "rationale": row["rationale"],
            "last_seen": row["last_seen"],
        }
    return result


def append_conversation(
    conn: sqlite3.Connection,
    session_id: int,
    concept_id: str | None,
    role: str,
    kind: str,
    content: str,
    payload: dict[str, Any] | None = None,
    *,
    timestamp: str | None = None,
) -> None:
    """Append one conversation entry to a session.

    Args:
        conn: Open database connection.
        session_id: Active session id.
        concept_id: Related concept, if any.
        role: ``system`` or ``student``.
        kind: Entry kind (``question``, ``response``, ``classification``, etc.).
        content: Primary text content.
        payload: Optional structured data (JSON-encoded before insert).
        timestamp: ISO timestamp; defaults to now.
    """
    payload_json = json.dumps(payload) if payload is not None else None
    conn.execute(
        """
        INSERT INTO conversation (
            session_id, concept_id, role, kind, content, payload, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            concept_id,
            role,
            kind,
            content,
            payload_json,
            timestamp or utc_now_iso(),
        ),
    )
    conn.commit()


def _conversation_count(conn: sqlite3.Connection, session_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM conversation WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row["n"])


def _load_needs_review(conn: sqlite3.Connection, student_id: str) -> list[str]:
    row = conn.execute(
        "SELECT needs_review FROM students WHERE id = ?",
        (student_id,),
    ).fetchone()
    if row is None:
        return []
    try:
        data = json.loads(row["needs_review"] or "[]")
    except json.JSONDecodeError:
        return []
    return list(data) if isinstance(data, list) else []


def student_profile_to_db(
    conn: sqlite3.Connection,
    profile: StudentProfile,
    session_id: int,
) -> None:
    """Persist a :class:`StudentProfile` to SQLite.

    Upserts the student row, all concept statuses, and any conversation
    entries not yet written for ``session_id`` (tracked by comparing the
    in-memory history length to the row count already stored).

    Args:
        conn: Open database connection.
        profile: In-session profile to flush.
        session_id: Active session receiving new conversation rows.
    """
    upsert_student(
        conn,
        profile.student_id,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
        needs_review=profile.needs_review,
    )

    for concept_id, record in profile.concept_statuses.items():
        diagnostic_label = str(record.get("status", ""))
        try:
            graph_status = diagnostic_to_graph_status(diagnostic_label)
        except ValueError:
            graph_status = str(record.get("status", "unassessed"))

        save_concept_status(
            conn,
            profile.student_id,
            concept_id,
            status=graph_status,
            diagnostic_label=diagnostic_label,
            confidence=float(record.get("confidence", 0.0)),
            misconception_summary=record.get("misconception_summary"),
            rationale=record.get("rationale"),
            last_seen=str(record.get("last_seen", utc_now_iso())),
        )

    persisted = _conversation_count(conn, session_id)
    new_entries = profile.conversation_history[persisted:]
    for entry in new_entries:
        payload = entry.get("result") or entry.get("payload")
        if payload is None and entry.get("kind") in {
            "classification",
            "understanding_check",
        }:
            payload = {k: v for k, v in entry.items() if k not in {"role", "kind", "content", "timestamp", "concept_id"}}
        append_conversation(
            conn,
            session_id,
            entry.get("concept_id"),
            str(entry.get("role", "system")),
            str(entry.get("kind", "message")),
            str(entry.get("content", "")),
            payload if isinstance(payload, dict) else None,
            timestamp=str(entry.get("timestamp", utc_now_iso())),
        )


def db_to_student_profile(
    conn: sqlite3.Connection,
    student_id: str,
    target_concept: str,
    *,
    include_conversation: bool = False,
    session_id: int | None = None,
) -> StudentProfile:
    """Reconstruct a :class:`StudentProfile` from SQLite.

    Args:
        conn: Open database connection.
        student_id: Student to load.
        target_concept: Target concept for the resumed session.
        include_conversation: When True, load conversation history from
            ``session_id`` or the student's most recent session.
        session_id: Specific session to load conversation from.

    Returns:
        A :class:`StudentProfile` populated from durable storage.
    """
    student_row = conn.execute(
        "SELECT created_at, updated_at, needs_review FROM students WHERE id = ?",
        (student_id,),
    ).fetchone()

    created_at = student_row["created_at"] if student_row else utc_now_iso()
    updated_at = student_row["updated_at"] if student_row else utc_now_iso()
    needs_review = _load_needs_review(conn, student_id)

    concept_statuses: dict[str, dict[str, Any]] = {}
    for concept_id, row in load_student_state(conn, student_id).items():
        diagnostic_label = row.get("diagnostic_label") or row.get("status")
        concept_statuses[concept_id] = {
            "status": diagnostic_label,
            "confidence": row.get("confidence", 0.0),
            "rationale": row.get("rationale"),
            "misconception_summary": row.get("misconception_summary"),
            "last_seen": row.get("last_seen", utc_now_iso()),
        }

    conversation_history: list[dict[str, Any]] = []
    if include_conversation:
        sid = session_id or _latest_session_id(conn, student_id)
        if sid is not None:
            conversation_history = _load_conversation(conn, sid)

    return StudentProfile(
        student_id=student_id,
        target_concept=target_concept,
        created_at=created_at,
        updated_at=updated_at,
        concept_statuses=concept_statuses,
        needs_review=needs_review,
        conversation_history=conversation_history,
    )


def _latest_session_id(conn: sqlite3.Connection, student_id: str) -> int | None:
    row = conn.execute(
        """
        SELECT id FROM sessions
        WHERE student_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (student_id,),
    ).fetchone()
    return int(row["id"]) if row else None


def _load_conversation(conn: sqlite3.Connection, session_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT concept_id, role, kind, content, payload, timestamp
        FROM conversation
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,),
    ).fetchall()

    history: list[dict[str, Any]] = []
    for row in rows:
        entry: dict[str, Any] = {
            "role": row["role"],
            "kind": row["kind"],
            "content": row["content"],
            "timestamp": row["timestamp"],
        }
        if row["concept_id"]:
            entry["concept_id"] = row["concept_id"]
        if row["payload"]:
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                if row["kind"] in {"classification", "understanding_check"}:
                    entry["result"] = payload
                else:
                    entry["payload"] = payload
        history.append(entry)
    return history


def table_names(conn: sqlite3.Connection) -> set[str]:
    """Return the set of user table names (for tests)."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {row["name"] for row in rows}
