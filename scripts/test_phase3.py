"""Phase 3 smoke test: diagnostic agent end-to-end against the local LLM.

Two modes:

* Default (canned). Runs three pre-written student responses against three
  concepts from the graph and prints the resulting DiagnosticResult for each.
  The canned responses are designed so a well-behaved 8B model produces one
  of each category: confirmed_known, confirmed_unknown, misconception_detected.
  Inspect the printed output and judge whether the classifications match.

* ``--interactive``. Generates a single Socratic question for the chosen
  concept and waits for your free-text response on stdin. Useful for
  feeling out how the agent behaves end-to-end before wiring it into a UI.

Prereqs:
    1. ``python scripts/test_phase2.py`` must pass first (Ollama up + model
       pulled).

Run:
    python scripts/test_phase3.py
    python scripts/test_phase3.py --interactive
    python scripts/test_phase3.py --interactive --concept recursion_basics
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from socratix.concept_graph import load_and_build  # noqa: E402
from socratix.diagnostic import (  # noqa: E402
    DiagnosticResult,
    classify_response,
    generate_question,
)
from socratix.llm import (  # noqa: E402
    DEFAULT_MODEL,
    OLLAMA_BASE_URL,
    OllamaError,
    is_ollama_running,
)


CANNED_SCENARIOS: list[tuple[str, str, str]] = [
    (
        "for_loops",
        "When I want to do the same thing for each item in a list, like printing every name. "
        "If I don't know how many times to repeat, I'd use a while loop instead.",
        "expected: confirmed_known",
    ),
    (
        "recursion_basics",
        "Honestly I have no idea what recursion is, I've never written one.",
        "expected: confirmed_unknown",
    ),
    (
        "base_case",
        "The base case is the first line of the function. It always runs before any other code, "
        "kind of like a setup step before the recursion starts.",
        "expected: misconception_detected",
    ),
]


def _fail(msg: str, hint: str | None = None) -> int:
    print(f"[FAIL] {msg}", file=sys.stderr)
    if hint:
        print(f"       hint: {hint}", file=sys.stderr)
    return 1


def _print_result(result: DiagnosticResult, *, expectation: str | None = None) -> None:
    if expectation:
        print(f"  [{expectation}]")
    print(f"  Q: {result.question}")
    print(f"  A: {result.student_response}")
    print(f"  -> classification: {result.classification}")
    print(f"     confidence:     {result.confidence:.2f}")
    print(f"     rationale:      {result.rationale}")
    if result.misconception_summary:
        print(f"     misconception:  {result.misconception_summary}")


def _run_canned() -> int:
    graph = load_and_build()
    print("\nRunning 3 canned scenarios (one per expected classification):")

    correct = 0
    for concept_id, student_response, expectation in CANNED_SCENARIOS:
        attrs = graph.nodes[concept_id]
        print(f"\n--- {concept_id}  ({attrs['name']}) ---")
        try:
            start = time.perf_counter()
            question = generate_question(
                concept_id, attrs["name"], attrs["description"]
            )
            elapsed_q = time.perf_counter() - start

            start = time.perf_counter()
            result = classify_response(
                concept_id, attrs["name"], question, student_response
            )
            elapsed_c = time.perf_counter() - start
        except OllamaError as exc:
            return _fail(f"LLM error during scenario {concept_id!r}: {exc}")

        _print_result(result, expectation=expectation)
        print(f"     (timing: question={elapsed_q:.1f}s, classify={elapsed_c:.1f}s)")

        expected_label = expectation.split(":", 1)[1].strip()
        if result.classification == expected_label:
            correct += 1

    print(
        f"\nScenarios matching expected classification: "
        f"{correct} / {len(CANNED_SCENARIOS)}"
    )
    print(
        "Note: with an 8B local model, 2/3 or 3/3 is a reasonable pass bar. "
        "If you see 0/3 or 1/3, tune the prompts in socratix/prompts.py before "
        "moving to Phase 4."
    )
    return 0


def _run_interactive(concept_id: str) -> int:
    graph = load_and_build()
    if concept_id not in graph:
        return _fail(
            f"Unknown concept id: {concept_id!r}",
            "Pick any id from data/concepts.json.",
        )
    attrs = graph.nodes[concept_id]

    print(f"\nConcept: {concept_id}  ({attrs['name']})")
    print(f"Description: {attrs['description']}\n")
    print("Generating a Socratic question (this can take a few seconds)...")
    try:
        question = generate_question(
            concept_id, attrs["name"], attrs["description"]
        )
    except OllamaError as exc:
        return _fail(f"LLM error generating the question: {exc}")

    print(f"\nQ: {question}\n")
    try:
        response = input("Your answer (Enter to submit): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        return 1
    if not response:
        response = "(student submitted no answer)"

    print("\nClassifying response...")
    try:
        result = classify_response(
            concept_id, attrs["name"], question, response
        )
    except OllamaError as exc:
        return _fail(f"LLM error classifying response: {exc}")

    print()
    _print_result(result)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 diagnostic agent smoke test."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate one question and wait for your free-text response.",
    )
    parser.add_argument(
        "--concept",
        default="for_loops",
        help="Concept id to use in --interactive mode (default: for_loops).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    print("Socratix Phase 3 - diagnostic agent smoke test")
    print("-" * 50)
    print(f"Ollama base URL : {OLLAMA_BASE_URL}")
    print(f"Default model   : {DEFAULT_MODEL}")
    print(f"Mode            : {'interactive' if args.interactive else 'canned'}")

    if not is_ollama_running():
        return _fail(
            f"Ollama is not responding on {OLLAMA_BASE_URL}.",
            "Run scripts/test_phase2.py first; it has detailed setup hints.",
        )

    if args.interactive:
        return _run_interactive(args.concept)
    return _run_canned()


if __name__ == "__main__":
    raise SystemExit(main())
