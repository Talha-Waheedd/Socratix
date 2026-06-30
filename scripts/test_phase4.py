"""Phase 4 smoke test: student profile persistence and graph sync.

Run from the repository root:

    python scripts/test_phase4.py

No LLM calls are made; this test uses synthetic DiagnosticResult objects.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from socratix.concept_graph import load_and_build  # noqa: E402
from socratix.diagnostic import DiagnosticResult  # noqa: E402
from socratix.student_model import (  # noqa: E402
    DEFAULT_PROFILES_DIR,
    apply_diagnostic_result,
    create_profile,
    diagnostic_to_graph_status,
    load_profile,
    profile_path,
    save_profile,
    sync_graph_from_profile,
)

STUDENT_ID = "test_student_001"
TARGET = "recursion_basics"


def _make_result(
    concept_id: str,
    classification: str,
    *,
    confidence: float = 0.9,
    rationale: str = "test rationale",
    misconception_summary: str | None = None,
    question: str | None = None,
    student_response: str | None = None,
) -> DiagnosticResult:
    return DiagnosticResult(
        concept_id=concept_id,
        classification=classification,  # type: ignore[arg-type]
        confidence=confidence,
        rationale=rationale,
        misconception_summary=misconception_summary,
        question=question or f"Socratic question about {concept_id}?",
        student_response=student_response or f"Student answer about {concept_id}.",
    )


def _profiles_equal(a, b) -> bool:
    return a.to_dict() == b.to_dict()


def main() -> int:
    print("Socratix Phase 4 - student model smoke test")
    print("-" * 50)

    graph = load_and_build()
    profile = create_profile(STUDENT_ID, TARGET)
    out_path = profile_path(STUDENT_ID, DEFAULT_PROFILES_DIR)

    results = [
        _make_result(
            "for_loops",
            "confirmed_known",
            confidence=0.92,
            rationale="Correctly distinguishes loop use cases.",
        ),
        _make_result(
            "recursion_basics",
            "confirmed_unknown",
            confidence=0.88,
            rationale="Student explicitly states unfamiliarity.",
        ),
        _make_result(
            "base_case",
            "misconception_detected",
            confidence=0.85,
            rationale="Conflates base case with function entry.",
            misconception_summary=(
                "Believes the base case is the first line executed."
            ),
        ),
    ]

    for result in results:
        apply_diagnostic_result(profile, graph, result)

    assert len(profile.conversation_history) == 9, (
        f"Expected 9 history entries, got {len(profile.conversation_history)}"
    )
    assert profile.needs_review == [], (
        f"Expected empty needs_review, got {profile.needs_review}"
    )

    for result in results:
        expected_graph = diagnostic_to_graph_status(result.classification)
        actual_graph = graph.nodes[result.concept_id]["status"]
        assert actual_graph == expected_graph, (
            f"{result.concept_id}: graph status {actual_graph!r} != {expected_graph!r}"
        )
        stored = profile.concept_statuses[result.concept_id]["status"]
        assert stored == result.classification, (
            f"{result.concept_id}: profile status {stored!r} != {result.classification!r}"
        )

    save_profile(profile, out_path)
    print(f"Saved profile to: {out_path}")

    loaded = load_profile(out_path)
    assert _profiles_equal(profile, loaded), "Round-trip profile mismatch"
    print("Round-trip load/save: OK")

    graph2 = load_and_build()
    sync_graph_from_profile(loaded, graph2)
    for result in results:
        expected = diagnostic_to_graph_status(result.classification)
        assert graph2.nodes[result.concept_id]["status"] == expected
    unassessed = sum(
        1 for _, attrs in graph2.nodes(data=True) if attrs["status"] == "unassessed"
    )
    assert unassessed == graph2.number_of_nodes() - len(results), (
        f"Expected {graph2.number_of_nodes() - len(results)} unassessed nodes"
    )
    print("sync_graph_from_profile: on fresh graph: OK")

    unclear = _make_result(
        "list_comprehension",
        "confirmed_unknown",
        confidence=0.0,
        rationale="Could not classify response after 3 attempts.",
    )
    apply_diagnostic_result(profile, graph, unclear)
    assert "list_comprehension" in profile.needs_review, (
        f"Expected list_comprehension in needs_review, got {profile.needs_review}"
    )
    print("confidence=0.0 -> needs_review: OK")

    print()
    print(f"Concept statuses recorded: {len(profile.concept_statuses)}")
    print(f"Conversation entries:      {len(profile.conversation_history)}")
    print(f"Needs review:              {profile.needs_review}")
    print()
    print("Phase 4 looks good. Ready to build the gap analyzer in Phase 5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
