"""Phase 8 smoke test: SQLite persistence layer.

Run from the repository root:

    python scripts/test_phase8.py

No LLM calls are made.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from socratix.concept_graph import load_and_build  # noqa: E402
from socratix.diagnostic import DiagnosticResult  # noqa: E402
from socratix.persistence import (  # noqa: E402
    append_conversation,
    db_to_student_profile,
    end_session,
    init_db,
    load_student_state,
    save_concept_status,
    start_session,
    student_profile_to_db,
    table_names,
    upsert_student,
)
from socratix.student_model import (  # noqa: E402
    apply_diagnostic_result,
    create_profile,
)


def _profiles_equivalent(a, b) -> bool:
    """Compare profiles ignoring exact timestamp strings on conversation rows."""
    if (
        a.student_id != b.student_id
        or a.target_concept != b.target_concept
        or a.needs_review != b.needs_review
        or a.concept_statuses != b.concept_statuses
    ):
        return False
    if len(a.conversation_history) != len(b.conversation_history):
        return False
    for left, right in zip(a.conversation_history, b.conversation_history):
        if left.get("role") != right.get("role"):
            return False
        if left.get("kind") != right.get("kind"):
            return False
        if left.get("content") != right.get("content"):
            return False
        if left.get("concept_id") != right.get("concept_id"):
            return False
    return True


def main() -> int:
    print("Socratix Phase 8 - SQLite persistence smoke test")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_socratix.sqlite"
        conn = init_db(db_path)

        tables = table_names(conn)
        for required in ("students", "sessions", "concept_status", "conversation"):
            assert required in tables, f"Missing table: {required}"
        print(f"Tables created: {sorted(tables)}")

        upsert_student(conn, "test_student_002")
        session_id = start_session(conn, "test_student_002", "recursion_basics")
        assert isinstance(session_id, int) and session_id > 0
        print(f"Started session id={session_id}: OK")

        save_concept_status(
            conn,
            "test_student_002",
            "for_loops",
            status="known",
            diagnostic_label="confirmed_known",
            confidence=0.9,
            misconception_summary=None,
            rationale="knows loops",
        )
        save_concept_status(
            conn,
            "test_student_002",
            "recursion_basics",
            status="unknown",
            diagnostic_label="confirmed_unknown",
            confidence=0.8,
            misconception_summary=None,
            rationale="unknown recursion",
        )
        save_concept_status(
            conn,
            "test_student_002",
            "base_case",
            status="misconception",
            diagnostic_label="misconception_detected",
            confidence=0.85,
            misconception_summary="Believes base case is first line.",
            rationale="misconception on base case",
        )

        entries = [
            ("for_loops", "system", "question", "When would you use a for loop?"),
            ("for_loops", "student", "response", "When iterating a list."),
            ("for_loops", "system", "classification", "confirmed_known"),
            ("recursion_basics", "system", "question", "What is recursion?"),
            ("recursion_basics", "student", "response", "I do not know."),
        ]
        for concept_id, role, kind, content in entries:
            append_conversation(
                conn,
                session_id,
                concept_id,
                role,
                kind,
                content,
                payload={"test": True} if kind == "classification" else None,
            )

        end_session(conn, session_id)
        conn.close()

        conn2 = init_db(db_path)
        state = load_student_state(conn2, "test_student_002")
        assert len(state) == 3
        assert state["for_loops"]["diagnostic_label"] == "confirmed_known"
        assert state["base_case"]["misconception_summary"] is not None
        print("load_student_state after reconnect: OK")

        graph = load_and_build()
        profile = create_profile("roundtrip_student", "recursion_basics")
        results = [
            DiagnosticResult(
                concept_id="variables",
                classification="confirmed_known",
                confidence=0.9,
                rationale="ok",
                question="Q?",
                student_response="A.",
            ),
            DiagnosticResult(
                concept_id="while_loops",
                classification="misconception_detected",
                confidence=0.8,
                rationale="wrong",
                misconception_summary="Thinks while always runs once.",
                question="Q2?",
                student_response="Always once.",
            ),
        ]
        for result in results:
            apply_diagnostic_result(profile, graph, result)
        profile.needs_review.append("while_loops")

        rt_session = start_session(conn2, profile.student_id, profile.target_concept)
        student_profile_to_db(conn2, profile, rt_session)

        loaded = db_to_student_profile(
            conn2,
            profile.student_id,
            profile.target_concept,
            include_conversation=True,
            session_id=rt_session,
        )
        assert loaded.concept_statuses == profile.concept_statuses
        assert loaded.needs_review == profile.needs_review
        assert _profiles_equivalent(profile, loaded)
        print("Profile round-trip (with conversation): OK")

        save_concept_status(
            conn2,
            profile.student_id,
            "variables",
            status="known",
            diagnostic_label="confirmed_known",
            confidence=0.99,
            misconception_summary=None,
            rationale="updated rationale",
        )
        updated = load_student_state(conn2, profile.student_id)["variables"]
        assert updated["confidence"] == 0.99
        assert updated["rationale"] == "updated rationale"
        print("Upsert same concept_id (UPDATE not INSERT): OK")

        conn2.close()

    print()
    print("Phase 8 looks good. Ready to build the Streamlit frontend in Phase 9.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
