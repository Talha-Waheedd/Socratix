"""Pyvis-based interactive visualization of the concept graph.

Nodes are colored by their per-student ``status`` attribute so the same
function can render both an unassessed reference graph and a live student
view in the Phase 9 Streamlit sidebar.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Final

import networkx as nx
from pyvis.network import Network

STATUS_COLORS: Final[dict[str, str]] = {
    "known": "#22c55e",
    "unknown": "#ef4444",
    "misconception": "#f97316",
    "unassessed": "#9ca3af",
}

_DEFAULT_COLOR: Final[str] = STATUS_COLORS["unassessed"]


def _node_tooltip(concept_id: str, attrs: dict) -> str:
    """Build an HTML tooltip for a node, safely escaping user-visible fields."""
    name = html.escape(str(attrs.get("name", concept_id)))
    description = html.escape(str(attrs.get("description", "")))
    category = html.escape(str(attrs.get("category", "")))
    status = html.escape(str(attrs.get("status", "unassessed")))
    prereqs = attrs.get("prerequisites", []) or []
    prereqs_str = html.escape(", ".join(prereqs)) if prereqs else "(none)"
    return (
        f"<b>{name}</b><br>"
        f"<i>{category}</i> &mdash; status: {status}<br><br>"
        f"{description}<br><br>"
        f"<b>Prerequisites:</b> {prereqs_str}"
    )


def visualize_graph(
    graph: nx.DiGraph,
    output_path: Path | str,
    *,
    height: str = "750px",
    width: str = "100%",
    open_browser: bool = False,
) -> Path:
    """Render the concept graph to a standalone HTML file with Pyvis.

    Args:
        graph: A validated concept :class:`networkx.DiGraph`.
        output_path: Where to write the HTML file. Parent directories are
            created if they do not exist.
        height: CSS height for the embedded network canvas.
        width: CSS width for the embedded network canvas.
        open_browser: If True, open the generated HTML in the default browser
            after writing. Defaults to False so test scripts stay headless.

    Returns:
        The absolute path of the written HTML file.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#ffffff",
        font_color="#111827",
        notebook=False,
        cdn_resources="remote",
    )
    # barnes_hut spreads ~50 nodes more legibly than the default force layout.
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=130,
        spring_strength=0.04,
        damping=0.09,
    )

    for node_id, attrs in graph.nodes(data=True):
        status = attrs.get("status", "unassessed")
        color = STATUS_COLORS.get(status, _DEFAULT_COLOR)
        net.add_node(
            node_id,
            label=attrs.get("name", node_id),
            title=_node_tooltip(node_id, attrs),
            color=color,
            shape="dot",
            size=18,
        )

    for source, target in graph.edges():
        net.add_edge(source, target, color="#94a3b8", arrows="to")

    # write_html avoids the implicit os.startfile that Network.show() triggers
    # on Windows when open_browser is undesired (e.g. in CI / test scripts).
    net.write_html(str(output_path), notebook=False, open_browser=open_browser)
    return output_path
