"""Minimum teaching path: unknown prerequisites between student and target.

Given a target concept and the student's current profile, :func:`find_gap_path`
returns the ordered list of concepts still needing coverage before the student
can reach the target.

Algorithm choice (documented here per project spec):

We use ``nx.ancestors(target) | {target}`` to collect every transitive
prerequisite of the target, filter to concepts the student has NOT yet
``confirmed_known``, then return them in topological order via
``nx.topological_sort`` on the induced subgraph.

Why not ``nx.shortest_path``?

- ``shortest_path`` returns ONE path between two specific nodes. Teaching
  requires every unknown prerequisite, possibly across multiple ancestor
  branches (e.g. ``binary_search`` depends on both ``linear_search`` and
  ``recursion_basics``).
- Topological sort on the induced subgraph guarantees each concept appears
  after all of its prerequisites — the correct teaching order automatically.
"""

from __future__ import annotations

import networkx as nx

from socratix.student_model import StudentProfile

_CONFIRMED_KNOWN = "confirmed_known"


def _needs_teaching(profile: StudentProfile, concept_id: str) -> bool:
    """Return True if the student still needs coverage for ``concept_id``.

    Unassessed concepts (absent from ``profile.concept_statuses``) count as
    needing teaching.
    """
    record = profile.concept_statuses.get(concept_id)
    if record is None:
        return True
    return record.get("status") != _CONFIRMED_KNOWN


def find_gap_path(
    graph: nx.DiGraph,
    profile: StudentProfile,
    target_concept_id: str,
) -> list[str]:
    """Return the minimum ordered teaching path toward a target concept.

    The path includes the target itself when it still needs teaching. Returns
    an empty list when the target is already ``confirmed_known`` and every
    prerequisite is likewise known.

    Args:
        graph: Validated concept :class:`networkx.DiGraph`.
        profile: Current student profile with ``concept_statuses``.
        target_concept_id: Concept id the student wants to learn.

    Returns:
        Concept ids in topological order (prerequisites before dependents).

    Raises:
        KeyError: If ``target_concept_id`` is not a node in ``graph``.
    """
    if target_concept_id not in graph:
        raise KeyError(f"Unknown concept: {target_concept_id!r}")

    candidates = nx.ancestors(graph, target_concept_id) | {target_concept_id}
    gap = {cid for cid in candidates if _needs_teaching(profile, cid)}

    if not gap:
        return []

    subgraph = graph.subgraph(gap)
    return list(nx.topological_sort(subgraph))


def summarize_gap(
    graph: nx.DiGraph,
    gap_path: list[str],
    *,
    target_concept_id: str | None = None,
) -> str:
    """Return a human-readable summary of the teaching gap for the UI sidebar.

    Args:
        graph: Concept graph (used to resolve display names).
        gap_path: Ordered list from :func:`find_gap_path`.
        target_concept_id: Optional override for the target display name.
            When omitted, the last item in ``gap_path`` is used if present.

    Returns:
        Multi-line plain-text summary suitable for Streamlit ``st.sidebar``.
    """
    if not gap_path:
        if target_concept_id and target_concept_id in graph:
            name = graph.nodes[target_concept_id].get("name", target_concept_id)
            return f"You already know everything needed to learn '{name}'."
        return "No concepts remain on the teaching path."

    target_id = target_concept_id or gap_path[-1]
    target_name = graph.nodes.get(target_id, {}).get("name", target_id)
    count = len(gap_path)

    labels = [
        graph.nodes[cid].get("name", cid) if cid in graph else cid
        for cid in gap_path
    ]
    ordered = ", ".join(labels)

    return (
        f"To learn '{target_name}' you need to cover {count} concept"
        f"{'s' if count != 1 else ''} in this order:\n{ordered}"
    )
