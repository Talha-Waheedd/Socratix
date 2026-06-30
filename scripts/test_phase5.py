"""Phase 5 smoke test: minimum teaching path via gap analysis.

Run from the repository root:

    python scripts/test_phase5.py

No LLM calls are made.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from socratix.concept_graph import load_and_build  # noqa: E402
from socratix.gap_analyzer import find_gap_path, summarize_gap  # noqa: E402
from socratix.student_model import create_profile, utc_now_iso  # noqa: E402

KNOWN_FOUNDATIONS = [
    "running_python",
    "variables",
    "data_types",
    "operators",
    "boolean_logic",
    "if_statements",
    "function_basics",
    "return_values",
]

TARGET = "recursion_basics"


def _mark_known(profile, concept_id: str) -> None:
    profile.concept_statuses[concept_id] = {
        "status": "confirmed_known",
        "confidence": 1.0,
        "rationale": "marked known for test",
        "misconception_summary": None,
        "last_seen": utc_now_iso(),
    }


def _assert_topological_order(graph, gap_path: list[str]) -> None:
    positions = {cid: i for i, cid in enumerate(gap_path)}
    for cid in gap_path:
        for prereq in graph.nodes[cid].get("prerequisites", []):
            if prereq in positions and positions[prereq] >= positions[cid]:
                raise AssertionError(
                    f"Topological order violated: {prereq!r} must appear "
                    f"before {cid!r} in gap path {gap_path}"
                )


def main() -> int:
    print("Socratix Phase 5 - gap analyzer smoke test")
    print("-" * 50)

    graph = load_and_build()
    profile = create_profile("gap_test_student", TARGET)

    for cid in KNOWN_FOUNDATIONS:
        _mark_known(profile, cid)

    gap = find_gap_path(graph, profile, TARGET)
    print(f"Gap toward {TARGET!r} (8 foundations known): {gap}")

    assert TARGET in gap, f"Expected target {TARGET!r} in gap, got {gap}"
    for known in KNOWN_FOUNDATIONS:
        assert known not in gap, f"Known concept {known!r} should not be in gap"
    _assert_topological_order(graph, gap)
    print("Includes target, excludes known prereqs, topological order: OK")

    summary = summarize_gap(graph, gap, target_concept_id=TARGET)
    assert "Recursion Basics" in summary or TARGET in summary
    print(f"Summary preview: {summary.splitlines()[0]}")

    prev_len = len(gap)
    _mark_known(profile, TARGET)
    gap_after = find_gap_path(graph, profile, TARGET)
    assert gap_after == [], f"Expected empty gap when target known, got {gap_after}"
    print("Target already known -> empty gap: OK")

    empty_summary = summarize_gap(graph, [], target_concept_id=TARGET)
    assert "already know" in empty_summary.lower()
    print("Empty gap summary: OK")

    profile2 = create_profile("gap_test_student_2", TARGET)
    for cid in KNOWN_FOUNDATIONS[:-1]:  # all except return_values
        _mark_known(profile2, cid)

    gap_before = find_gap_path(graph, profile2, TARGET)
    _mark_known(profile2, "return_values")
    gap_after_more_known = find_gap_path(graph, profile2, TARGET)
    assert len(gap_after_more_known) < len(gap_before), (
        f"Gap should shrink when marking return_values known: "
        f"{len(gap_before)} -> {len(gap_after_more_known)}"
    )
    print("Monotonic shrink check: OK")

    try:
        find_gap_path(graph, profile, "not_a_real_concept")
        print("[FAIL] Expected KeyError for unknown target", file=sys.stderr)
        return 1
    except KeyError:
        print("Unknown target -> KeyError: OK")

    binary_gap = find_gap_path(
        graph, create_profile("binary_test", "binary_search"), "binary_search"
    )
    assert "binary_search" in binary_gap
    _assert_topological_order(graph, binary_gap)
    assert (
        "linear_search" in binary_gap or "recursion_basics" in binary_gap
    ), "binary_search gap should include at least one branch prerequisite"
    print(f"Multi-branch gap for binary_search ({len(binary_gap)} concepts): OK")

    print()
    print("Phase 5 looks good. Ready to build the misconception database in Phase 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
