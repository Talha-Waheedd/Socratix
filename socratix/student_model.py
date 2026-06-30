"""Per-student knowledge state: JSON profiles synced with the concept graph.

A :class:`StudentProfile` tracks which concepts are known, unknown, or carry
a detected misconception, plus the full diagnostic conversation history.
Profiles are saved as ``student_profiles/<student_id>.json`` and are the
in-session working copy; Phase 8 adds SQLite for cross-day persistence.

The profile stores diagnostic labels (``confirmed_known``) while the NetworkX
graph stores short visualization labels (``known``). Use
:func:`diagnostic_to_graph_status` and :func:`graph_status_to_diagnostic`
to translate between them.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from socratix.concept_graph import set_node_status
from socratix.diagnostic import Classification, DiagnosticResult

logger = logging.getLogger(__name__)

DEFAULT_PROFILES_DIR: Path = (
    Path(__file__).resolve().parent.parent / "student_profiles"
)

# Diagnostic label (profile) -> graph visualization label
_DIAGNOSTIC_TO_GRAPH: dict[str, str] = {
    "confirmed_known": "known",
    "confirmed_unknown": "unknown",
    "misconception_detected": "misconception",
}

_GRAPH_TO_DIAGNOSTIC: dict[str, str] = {
    "known": "confirmed_known",
    "unknown": "confirmed_unknown",
    "misconception": "misconception_detected",
}


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def diagnostic_to_graph_status(classification: Classification | str) -> str:
    """Map a diagnostic classification to a graph node status label.

    Args:
        classification: One of ``confirmed_known``, ``confirmed_unknown``,
            ``misconception_detected``.

    Returns:
        One of ``known``, ``unknown``, ``misconception``.

    Raises:
        ValueError: If ``classification`` is not a recognized diagnostic label.
    """
    try:
        return _DIAGNOSTIC_TO_GRAPH[classification]
    except KeyError as exc:
        raise ValueError(
            f"Unknown diagnostic classification: {classification!r}"
        ) from exc


def graph_status_to_diagnostic(graph_status: str) -> str | None:
    """Map a graph node status back to a diagnostic classification label.

    Args:
        graph_status: One of ``known``, ``unknown``, ``misconception``,
            or ``unassessed``.

    Returns:
        The corresponding diagnostic label, or ``None`` for ``unassessed``.
    """
    if graph_status == "unassessed":
        return None
    return _GRAPH_TO_DIAGNOSTIC.get(graph_status)


def profile_path(
    student_id: str,
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
) -> Path:
    """Return the JSON file path for a student's profile."""
    return profiles_dir / f"{student_id}.json"


@dataclass
class StudentProfile:
    """Mutable per-student session state.

    Attributes:
        student_id: Unique identifier for the student.
        created_at: ISO 8601 UTC timestamp of profile creation.
        updated_at: ISO 8601 UTC timestamp of the last mutation.
        target_concept: The concept id the student wants to learn.
        concept_statuses: Map of concept_id -> diagnostic record dict.
            Only assessed concepts appear here; unassessed concepts are
            omitted to keep profiles small.
        needs_review: Concept ids flagged for human review (classification
            failures, teaching retries exhausted, etc.).
        conversation_history: Chronological list of dialogue entries.
    """

    student_id: str
    target_concept: str
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    concept_statuses: dict[str, dict[str, Any]] = field(default_factory=dict)
    needs_review: list[str] = field(default_factory=list)
    conversation_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile to a JSON-compatible dict."""
        return {
            "student_id": self.student_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "target_concept": self.target_concept,
            "concept_statuses": self.concept_statuses,
            "needs_review": list(self.needs_review),
            "conversation_history": self.conversation_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StudentProfile:
        """Reconstruct a profile from a parsed JSON dict."""
        return cls(
            student_id=data["student_id"],
            target_concept=data["target_concept"],
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
            concept_statuses=dict(data.get("concept_statuses", {})),
            needs_review=list(data.get("needs_review", [])),
            conversation_history=list(data.get("conversation_history", [])),
        )


def create_profile(student_id: str, target_concept: str) -> StudentProfile:
    """Create a fresh profile with empty assessment state.

    No entries are written to ``concept_statuses``; every concept in the
    graph remains implicitly ``unassessed`` until a diagnostic result is
    applied.

    Args:
        student_id: Unique student identifier.
        target_concept: Concept id the student wants to reach.

    Returns:
        A new :class:`StudentProfile`.
    """
    now = utc_now_iso()
    return StudentProfile(
        student_id=student_id,
        target_concept=target_concept,
        created_at=now,
        updated_at=now,
    )


def load_profile(path: Path | str) -> StudentProfile:
    """Load a student profile from a JSON file.

    Args:
        path: Path to the profile JSON.

    Returns:
        A :class:`StudentProfile` reconstructed from disk.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is missing required keys.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Profile not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    for key in ("student_id", "target_concept"):
        if key not in data:
            raise ValueError(f"Profile JSON missing required key: {key!r}")

    return StudentProfile.from_dict(data)


def save_profile(profile: StudentProfile, path: Path | str) -> None:
    """Atomically persist a profile to disk.

    Writes to a temporary file in the same directory, then renames it over
    the target path so a crash mid-write never leaves a half-written profile.

    Args:
        profile: The profile to save.
        path: Destination JSON path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(profile.to_dict(), indent=2, ensure_ascii=False)
    tmp_path.write_text(payload + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def mark_for_review(
    profile: StudentProfile,
    concept_id: str,
    reason: str,
) -> None:
    """Flag a concept for human review.

    The ``reason`` is logged for debugging; only ``concept_id`` is stored
    in ``profile.needs_review`` (per the JSON schema).

    Args:
        profile: The student profile to update.
        concept_id: Concept that could not be classified or taught reliably.
        reason: Human-readable explanation (logged, not persisted).
    """
    if concept_id not in profile.needs_review:
        profile.needs_review.append(concept_id)
        logger.info(
            "Concept %r flagged for review (student=%s): %s",
            concept_id,
            profile.student_id,
            reason,
        )


def apply_diagnostic_result(
    profile: StudentProfile,
    graph: nx.DiGraph,
    result: DiagnosticResult,
) -> None:
    """Apply a diagnostic classification to the profile and concept graph.

    Updates ``profile.concept_statuses``, the graph node's ``status``
    attribute, and ``profile.conversation_history`` (three entries per
    turn: question, response, classification). If ``result.confidence``
    is ``0.0`` (classification retries exhausted), also flags the concept
    for human review.

    Args:
        profile: Mutable student profile.
        graph: Concept graph whose node ``status`` attrs will be updated.
        result: Structured outcome from :func:`socratix.diagnostic.classify_response`.

    Raises:
        KeyError: If ``result.concept_id`` is not a node in ``graph``.
    """
    if result.concept_id not in graph:
        raise KeyError(f"Unknown concept id: {result.concept_id!r}")

    now = utc_now_iso()
    graph_status = diagnostic_to_graph_status(result.classification)

    profile.concept_statuses[result.concept_id] = {
        "status": result.classification,
        "confidence": result.confidence,
        "rationale": result.rationale,
        "misconception_summary": result.misconception_summary,
        "last_seen": now,
    }

    set_node_status(graph, result.concept_id, graph_status)

    if result.question:
        profile.conversation_history.append(
            {
                "role": "system",
                "kind": "question",
                "concept_id": result.concept_id,
                "content": result.question,
                "timestamp": now,
            }
        )
    if result.student_response is not None:
        profile.conversation_history.append(
            {
                "role": "student",
                "kind": "response",
                "concept_id": result.concept_id,
                "content": result.student_response,
                "timestamp": now,
            }
        )
    profile.conversation_history.append(
        {
            "role": "system",
            "kind": "classification",
            "concept_id": result.concept_id,
            "content": result.classification,
            "result": result.to_dict(),
            "timestamp": now,
        }
    )

    if result.confidence == 0.0:
        mark_for_review(
            profile,
            result.concept_id,
            reason=result.rationale or "Classification confidence was 0.0",
        )

    profile.updated_at = now


def sync_graph_from_profile(
    profile: StudentProfile,
    graph: nx.DiGraph,
) -> None:
    """Restore graph node statuses from a loaded profile.

    Resets every node to ``unassessed`` first, then applies each entry in
    ``profile.concept_statuses``. Concept ids present in the profile but
    absent from the graph (e.g. after editing ``concepts.json``) are
    skipped with a warning.

    Args:
        profile: Profile whose ``concept_statuses`` drive the update.
        graph: Concept graph to mutate in place.
    """
    for node_id in graph.nodes:
        set_node_status(graph, node_id, "unassessed")

    for concept_id, record in profile.concept_statuses.items():
        if concept_id not in graph:
            warnings.warn(
                f"Profile references unknown concept {concept_id!r}; skipping.",
                stacklevel=2,
            )
            continue

        diagnostic_label = record.get("status")
        if diagnostic_label is None:
            continue

        try:
            graph_status = diagnostic_to_graph_status(diagnostic_label)
        except ValueError:
            warnings.warn(
                f"Profile has invalid status {diagnostic_label!r} for "
                f"{concept_id!r}; skipping.",
                stacklevel=2,
            )
            continue

        set_node_status(graph, concept_id, graph_status)
