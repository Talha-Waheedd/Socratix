"""Phase 1 smoke test: build the concept graph and render the Pyvis HTML.

Run from the repository root:

    python scripts/test_phase1.py

On success the script exits with status 0 and writes an interactive HTML
visualization to ``output/concept_graph.html`` that you can open in any
browser to inspect the graph manually.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from socratix.concept_graph import (  # noqa: E402  (sys.path tweak above)
    MAX_NODES,
    MIN_NODES,
    get_graph_stats,
    load_and_build,
    set_node_status,
)
from socratix.visualize import visualize_graph  # noqa: E402


SAMPLE_STATUSES: dict[str, str] = {
    "running_python": "known",
    "variables": "known",
    "data_types": "known",
    "for_loops": "misconception",
    "recursion_basics": "unknown",
    "list_comprehension": "unknown",
}


def main() -> int:
    graph = load_and_build()

    stats = get_graph_stats(graph)
    assert MIN_NODES <= stats["num_nodes"] <= MAX_NODES, (
        f"Node count {stats['num_nodes']} outside [{MIN_NODES}, {MAX_NODES}]"
    )
    assert stats["is_dag"], "Concept graph must be a DAG."

    for concept_id, status in SAMPLE_STATUSES.items():
        set_node_status(graph, concept_id, status)

    output_path = REPO_ROOT / "output" / "concept_graph.html"
    html_path = visualize_graph(graph, output_path)

    updated_stats = get_graph_stats(graph)

    print("Socratix Phase 1 - concept graph smoke test")
    print("-" * 50)
    print(f"Nodes: {updated_stats['num_nodes']}")
    print(f"Edges: {updated_stats['num_edges']}")
    print(f"DAG:   {'OK' if updated_stats['is_dag'] else 'FAIL'}")
    print()
    print("Nodes per category:")
    for category, count in sorted(updated_stats["categories"].items()):
        print(f"  {category:<15} {count}")
    print()
    print("Sample status assignments (for color verification):")
    for concept_id, status in SAMPLE_STATUSES.items():
        print(f"  {concept_id:<22} -> {status}")
    print()
    print(f"HTML written to: {html_path}")
    print("Open that file in a browser to inspect the graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
