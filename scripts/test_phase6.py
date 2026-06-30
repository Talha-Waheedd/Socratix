"""Phase 6 smoke test: ChromaDB misconception retrieval.

Run from the repository root:

    python scripts/test_phase6.py

First run downloads the sentence-transformers model (~80 MB). This is
expected and only happens once.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from socratix.misconceptions import (  # noqa: E402
    DEFAULT_MISCONCEPTIONS_PATH,
    build_misconception_db,
    find_correction,
    get_collection,
    reset_persist_dir,
)

TEST_PERSIST_DIR = REPO_ROOT / "chroma_db_test"
EXPECTED_COUNT = 20


def main() -> int:
    print("Socratix Phase 6 - misconception database smoke test")
    print("-" * 50)
    print(f"Seed data: {DEFAULT_MISCONCEPTIONS_PATH}")
    print(f"Test DB dir: {TEST_PERSIST_DIR}")
    print()
    print("Building Chroma DB (first run may download ~80 MB model)...")

    reset_persist_dir(TEST_PERSIST_DIR)
    collection = build_misconception_db(
        DEFAULT_MISCONCEPTIONS_PATH,
        persist_dir=TEST_PERSIST_DIR,
    )
    assert collection.count() == EXPECTED_COUNT, (
        f"Expected {EXPECTED_COUNT} documents, got {collection.count()}"
    )
    print(f"Seeded {collection.count()} misconceptions: OK")

    match1 = find_correction(
        collection,
        "I think recursion needs a for loop inside it to repeat.",
    )
    assert match1 is not None, "Expected a match for recursion + loop misconception"
    assert match1["entry_id"] == "recursion_loop_inside", (
        f"Expected recursion_loop_inside, got {match1['entry_id']!r}"
    )
    assert match1["distance"] < 0.5, (
        f"Expected distance < 0.5, got {match1['distance']:.3f}"
    )
    print(
        f"Query 1 -> {match1['entry_id']} (distance={match1['distance']:.3f}): OK"
    )

    match2 = find_correction(
        collection,
        "I believe list indices cannot be negative in Python.",
    )
    assert match2 is not None, "Expected a match for negative index misconception"
    assert match2["entry_id"] == "negative_index_error", (
        f"Expected negative_index_error, got {match2['entry_id']!r}"
    )
    print(
        f"Query 2 -> {match2['entry_id']} (distance={match2['distance']:.3f}): OK"
    )

    match3 = find_correction(
        collection,
        "What is the syntax of a class definition in Python?",
    )
    assert match3 is None, (
        f"Expected no match for unrelated query, got {match3!r}"
    )
    print("Query 3 (unrelated) -> no match: OK")

    reopened = get_collection(persist_dir=TEST_PERSIST_DIR)
    assert reopened.count() == EXPECTED_COUNT, (
        f"Persistence check failed: {reopened.count()} docs"
    )
    print(f"Re-opened collection has {reopened.count()} docs: OK")

    print()
    print("Sample correction (recursion):")
    print(f"  {match1['correction'][:120]}...")
    print()
    print("Phase 6 looks good. Ready to build the teaching agent in Phase 7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
