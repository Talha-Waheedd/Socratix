"""Phase 7 smoke test: teaching agent with local LLM.

Run from the repository root:

    python scripts/test_phase7.py

Requires Ollama (same setup as Phase 2). Uses stubbed student responses
so the test is deterministic; the LLM generates explanations and judges
understanding.

    python scripts/test_phase7.py --quick

Uses stubbed LLM calls for faster logic verification when Ollama is slow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from socratix.concept_graph import load_and_build  # noqa: E402
from socratix.llm import OLLAMA_BASE_URL, is_ollama_running  # noqa: E402
from socratix.misconceptions import (  # noqa: E402
    DEFAULT_MISCONCEPTIONS_PATH,
    build_misconception_db,
    reset_persist_dir,
)
from socratix.student_model import create_profile, utc_now_iso  # noqa: E402
from socratix.teaching_agent import (  # noqa: E402
    MAX_TEACHING_RETRIES,
    check_understanding,
    generate_explanation,
    teach_concept,
)

TEST_CHROMA_DIR = REPO_ROOT / "chroma_db_test_phase7"
KNOWN_IDS = [
    "running_python",
    "variables",
    "data_types",
    "function_basics",
    "return_values",
    "if_statements",
]
TARGET = "recursion_basics"


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}", file=sys.stderr)
    return 1


def _mark_known(profile, graph, concept_id: str) -> None:
    profile.concept_statuses[concept_id] = {
        "status": "confirmed_known",
        "confidence": 1.0,
        "rationale": "marked known for test",
        "misconception_summary": None,
        "last_seen": utc_now_iso(),
    }
    graph.nodes[concept_id]["status"] = "known"


def _stub_explanation(*_args, **_kwargs) -> str:
    return (
        "Think of recursion like nested Russian dolls — each doll contains a "
        "smaller one until you reach the smallest (the base case).\n\n"
        "```python\n"
        "def countdown(n):\n"
        "    if n <= 0:\n"
        "        return\n"
        "    print(n)\n"
        "    countdown(n - 1)\n"
        "```\n\n"
        "What would countdown(3) print?"
    )


def _stub_understood(*_args, **_kwargs) -> dict:
    return {
        "verdict": "understood",
        "confidence": 0.9,
        "rationale": "Student applied the concept correctly.",
    }


def _stub_struggling(*_args, **_kwargs) -> dict:
    return {
        "verdict": "still_struggling",
        "confidence": 0.85,
        "rationale": "Student still confused.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7 teaching agent test.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Stub LLM calls; verify teach_concept logic only.",
    )
    args = parser.parse_args()

    print("Socratix Phase 7 - teaching agent smoke test")
    print("-" * 50)
    print(f"Ollama base URL : {OLLAMA_BASE_URL}")
    print(f"Mode            : {'quick (stubbed LLM)' if args.quick else 'live LLM'}")

    if not args.quick and not is_ollama_running():
        return _fail(
            "Ollama is not running. Start it or use --quick for offline logic checks."
        )

    graph = load_and_build()
    profile = create_profile("phase7_test_student", TARGET)
    for cid in KNOWN_IDS:
        _mark_known(profile, graph, cid)

    reset_persist_dir(TEST_CHROMA_DIR)
    collection = build_misconception_db(
        DEFAULT_MISCONCEPTIONS_PATH,
        persist_dir=TEST_CHROMA_DIR,
    )

    patches = []
    if args.quick:
        patches = [
            patch(
                "socratix.teaching_agent.generate_explanation",
                side_effect=_stub_explanation,
            ),
            patch(
                "socratix.teaching_agent.check_understanding",
                side_effect=_stub_understood,
            ),
        ]

    print("\n[1/2] Success path (student demonstrates understanding)...")
    if patches:
        for p in patches:
            p.start()
    try:
        result_ok = teach_concept(
            graph,
            profile,
            TARGET,
            collection,
            student_response_provider=lambda _exp: (
                "Recursion means a function calls itself on a smaller input "
                "until a base case stops it. countdown(3) would print 3, 2, 1."
            ),
            max_retries=MAX_TEACHING_RETRIES,
        )
    finally:
        if patches:
            for p in patches:
                p.stop()

    if not result_ok.success:
        return _fail(
            f"Expected teaching success, got {result_ok!r}. "
            "Re-run with --quick or tune prompts in socratix/prompts.py."
        )
    assert result_ok.retries_used == 0, (
        f"Expected retries_used=0, got {result_ok.retries_used}"
    )
    assert profile.concept_statuses[TARGET]["status"] == "confirmed_known"
    assert graph.nodes[TARGET]["status"] == "known"
    print(f"      success=True, retries_used={result_ok.retries_used}: OK")

    profile2 = create_profile("phase7_fail_student", TARGET)
    graph2 = load_and_build()
    for cid in KNOWN_IDS:
        _mark_known(profile2, graph2, cid)

    fail_patches = []
    if args.quick:
        fail_patches = [
            patch(
                "socratix.teaching_agent.generate_explanation",
                side_effect=_stub_explanation,
            ),
            patch(
                "socratix.teaching_agent.check_understanding",
                side_effect=_stub_struggling,
            ),
        ]

    print("\n[2/2] Failure path (student never understands -> flagged)...")
    if fail_patches:
        for p in fail_patches:
            p.start()
    try:
        result_fail = teach_concept(
            graph2,
            profile2,
            TARGET,
            collection,
            student_response_provider=lambda _exp: "I still do not get it at all.",
            max_retries=MAX_TEACHING_RETRIES,
        )
    finally:
        if fail_patches:
            for p in fail_patches:
                p.stop()

    if result_fail.success:
        return _fail("Expected teaching failure after max retries.")
    assert result_fail.retries_used == MAX_TEACHING_RETRIES
    assert result_fail.flagged_for_review is True
    assert TARGET in profile2.needs_review
    assert len(result_fail.transcript) == MAX_TEACHING_RETRIES + 1
    print(
        f"      success=False, retries_used={result_fail.retries_used}, "
        f"transcript rounds={len(result_fail.transcript)}: OK"
    )

    if not args.quick:
        print("\n[bonus] Live generate_explanation snippet:")
        attrs = graph.nodes[TARGET]
        snippet = generate_explanation(
            attrs,
            ["Variables and Assignment", "For Loops"],
            None,
        )
        print(snippet[:280] + ("..." if len(snippet) > 280 else ""))

    print()
    print("Phase 7 looks good. Ready to build SQLite persistence in Phase 8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
