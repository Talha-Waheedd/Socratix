"""Concept graph construction, validation, and per-student status tracking.

The concept graph is a directed acyclic graph (DAG) of Python-fundamentals
concepts loaded from an editable JSON file. Edges point from prerequisite to
dependent concept (e.g. ``variables -> data_types``), so a topological sort
yields a valid teaching order. Later phases attach a per-student ``status``
to each node (``known``, ``unknown``, ``misconception``, ``unassessed``);
the visualizer reads that attribute to color nodes.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx

DEFAULT_CONCEPTS_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "concepts.json"

VALID_STATUSES: frozenset[str] = frozenset(
    {"known", "unknown", "misconception", "unassessed"}
)

MIN_NODES: int = 40
MAX_NODES: int = 60

_REQUIRED_CONCEPT_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "description",
    "category",
    "prerequisites",
)


def load_concepts(path: Path | str = DEFAULT_CONCEPTS_PATH) -> dict[str, Any]:
    """Load and lightly validate the concepts JSON file.

    Args:
        path: Path to the concepts JSON. Defaults to ``data/concepts.json``
            relative to the package root.

    Returns:
        Parsed JSON document as a dict with keys ``version``, ``domain``,
        and ``concepts``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is missing required top-level keys or any
            concept entry is missing required fields.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Concepts file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    if "concepts" not in data or not isinstance(data["concepts"], list):
        raise ValueError("Concepts JSON must contain a top-level 'concepts' list.")

    for i, concept in enumerate(data["concepts"]):
        missing = [field for field in _REQUIRED_CONCEPT_FIELDS if field not in concept]
        if missing:
            raise ValueError(
                f"Concept at index {i} is missing required fields: {missing}"
            )
        if not isinstance(concept["prerequisites"], list):
            raise ValueError(
                f"Concept '{concept['id']}' has non-list prerequisites."
            )

    return data


def build_graph(concepts_data: dict[str, Any]) -> nx.DiGraph:
    """Build a NetworkX DiGraph from a parsed concepts document.

    Each node receives the attributes ``name``, ``description``, ``category``,
    and ``status`` (defaulting to ``"unassessed"``). Edges go from each
    prerequisite to its dependent concept.

    Args:
        concepts_data: The dict returned by :func:`load_concepts`.

    Returns:
        A directed graph whose nodes are concept ids.
    """
    graph: nx.DiGraph = nx.DiGraph()

    for concept in concepts_data["concepts"]:
        graph.add_node(
            concept["id"],
            name=concept["name"],
            description=concept["description"],
            category=concept["category"],
            prerequisites=list(concept["prerequisites"]),
            status="unassessed",
        )

    for concept in concepts_data["concepts"]:
        for prereq_id in concept["prerequisites"]:
            graph.add_edge(prereq_id, concept["id"])

    return graph


def validate_graph(graph: nx.DiGraph) -> None:
    """Validate structural invariants of the concept graph.

    Checks performed:
        * Every prerequisite referenced by an edge corresponds to a known node.
        * The graph contains between :data:`MIN_NODES` and :data:`MAX_NODES`
          nodes (inclusive).
        * The graph is a DAG (no prerequisite cycles).

    Args:
        graph: The graph returned by :func:`build_graph`.

    Raises:
        ValueError: If any invariant is violated. The message identifies the
            specific problem so editing ``concepts.json`` is straightforward.
    """
    node_ids: set[str] = set(graph.nodes())

    dangling: list[tuple[str, str]] = []
    for node_id, attrs in graph.nodes(data=True):
        for prereq in attrs.get("prerequisites", []):
            if prereq not in node_ids:
                dangling.append((node_id, prereq))
    if dangling:
        details = ", ".join(f"{dep!r} requires unknown {prereq!r}" for dep, prereq in dangling)
        raise ValueError(f"Dangling prerequisite references: {details}")

    n_nodes: int = graph.number_of_nodes()
    if not (MIN_NODES <= n_nodes <= MAX_NODES):
        raise ValueError(
            f"Concept graph has {n_nodes} nodes; expected between "
            f"{MIN_NODES} and {MAX_NODES} (inclusive)."
        )

    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph, orientation="original")
        raise ValueError(f"Concept graph contains a cycle: {cycle}")


def set_node_status(graph: nx.DiGraph, concept_id: str, status: str) -> None:
    """Update the per-student status attribute on a node.

    Args:
        graph: The concept graph.
        concept_id: The node id whose status should change.
        status: One of :data:`VALID_STATUSES`.

    Raises:
        KeyError: If ``concept_id`` is not a node in the graph.
        ValueError: If ``status`` is not a recognized value.
    """
    if concept_id not in graph:
        raise KeyError(f"Unknown concept id: {concept_id!r}")
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status {status!r}; expected one of {sorted(VALID_STATUSES)}"
        )
    graph.nodes[concept_id]["status"] = status


def get_graph_stats(graph: nx.DiGraph) -> dict[str, Any]:
    """Return a small summary dict useful for tests and the Streamlit sidebar.

    Args:
        graph: The concept graph.

    Returns:
        A dict containing node count, edge count, DAG flag, per-category node
        counts, and per-status node counts.
    """
    category_counts: Counter[str] = Counter(
        attrs.get("category", "uncategorized") for _, attrs in graph.nodes(data=True)
    )
    status_counts: Counter[str] = Counter(
        attrs.get("status", "unassessed") for _, attrs in graph.nodes(data=True)
    )
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "is_dag": nx.is_directed_acyclic_graph(graph),
        "categories": dict(category_counts),
        "statuses": dict(status_counts),
    }


def load_and_build(path: Path | str = DEFAULT_CONCEPTS_PATH) -> nx.DiGraph:
    """Convenience helper: load, build, validate, and return the graph.

    Args:
        path: Optional override for the concepts JSON path.

    Returns:
        A validated :class:`networkx.DiGraph`.
    """
    data = load_concepts(path)
    graph = build_graph(data)
    validate_graph(graph)
    return graph
