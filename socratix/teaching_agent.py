"""Phase 7 teaching agent: explanations, understanding checks, retry logic.

Generates concept explanations calibrated to what the student already knows
(analogies) and any misconception correction retrieved from ChromaDB. After
each explanation the student must apply the concept; if they still struggle,
the agent retries once with a different framing (max 2 retries total), then
flags the concept for human review.

Local-LLM tradeoff: 8B models often produce shallow or off-topic analogies.
The retry path mitigates this once; after that, :func:`teach_concept` stops
rather than looping forever.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx
from chromadb import Collection

from socratix.concept_graph import set_node_status
from socratix.llm import DEFAULT_MODEL, OllamaError, ask_llm, ask_llm_json
from socratix.misconceptions import find_correction
from socratix.prompts import (
    TEACHING_EXPLANATION,
    TEACHING_RETRY_EXPLANATION,
    UNDERSTANDING_CHECK,
    UNDERSTANDING_CHECK_STRICT,
)
from socratix.student_model import (
    StudentProfile,
    diagnostic_to_graph_status,
    mark_for_review,
    utc_now_iso,
)

UnderstandingVerdict = Literal["understood", "still_struggling"]

_VALID_VERDICTS: frozenset[str] = frozenset({"understood", "still_struggling"})

MAX_TEACHING_RETRIES: int = 2
"""Maximum retry count after the first explanation (3 attempts total: 0, 1, 2)."""

MAX_UNDERSTANDING_JSON_RETRIES: int = 2
_EXPLANATION_TEMPERATURE: float = 0.5
_UNDERSTANDING_TEMPERATURE_FIRST: float = 0.2
_UNDERSTANDING_TEMPERATURE_RETRY: float = 0.1

_FALLBACK_ANALOGY_CONCEPTS: tuple[str, ...] = (
    "Variables and Assignment",
    "Running Python",
)


@dataclass
class TeachResult:
    """Outcome of teaching one concept through explanation + follow-up checks.

    Attributes:
        concept_id: The concept that was taught.
        success: True if the student demonstrated understanding within the
            retry budget.
        retries_used: Number of retries consumed (0 on first-attempt success,
            up to :data:`MAX_TEACHING_RETRIES` on failure).
        flagged_for_review: True when all attempts failed and the concept was
            flagged for human review.
        transcript: Chronological teaching trail (explanation, followup,
            verdict dicts per attempt).
    """

    concept_id: str
    success: bool
    retries_used: int
    flagged_for_review: bool
    transcript: list[dict[str, Any]] = field(default_factory=list)


def _known_concept_names(
    graph: nx.DiGraph,
    profile: StudentProfile,
) -> list[str]:
    """Return display names of concepts the student has confirmed they know."""
    names: list[str] = []
    for concept_id, record in profile.concept_statuses.items():
        if record.get("status") != "confirmed_known":
            continue
        if concept_id in graph:
            names.append(graph.nodes[concept_id].get("name", concept_id))
    if not names:
        return list(_FALLBACK_ANALOGY_CONCEPTS)
    return names


def _guess_analogy_from_explanation(explanation: str) -> str | None:
    """Best-effort extraction of an analogy concept name from prose."""
    patterns = [
        r"analogy (?:to|with|from) ([^.?\n]+)",
        r"like ([^.?\n]+)",
        r"similar to ([^.?\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, explanation, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def generate_explanation(
    concept_attrs: dict[str, Any],
    known_concept_names: list[str],
    misconception_correction: str | None,
    *,
    retry_n: int = 0,
    previous_analogy: str | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate a teaching explanation for one concept.

    Args:
        concept_attrs: Graph node attributes (``name``, ``description``, etc.).
        known_concept_names: Concepts to draw analogies from.
        misconception_correction: Optional correction text from ChromaDB.
        retry_n: 0 for first attempt; >= 1 triggers the retry prompt with a
            different analogy instruction.
        previous_analogy: Analogy used in the prior attempt (retry path only).
        model: Ollama model tag.

    Returns:
        The full explanation string including code block and application
        question.

    Raises:
        OllamaError: If the local LLM is unavailable.
    """
    system_prompt = TEACHING_EXPLANATION
    if retry_n >= 1:
        system_prompt = f"{TEACHING_EXPLANATION}\n\n{TEACHING_RETRY_EXPLANATION}"

    known_str = ", ".join(known_concept_names) if known_concept_names else "(none)"
    correction_block = (
        misconception_correction
        if misconception_correction
        else "None — no specific misconception correction for this concept."
    )

    user_parts = [
        f"Concept: {concept_attrs.get('name', 'Unknown')}",
        f"Description: {concept_attrs.get('description', '')}",
        f"Known concepts for analogies: {known_str}",
        f"Misconception correction: {correction_block}",
    ]
    if retry_n >= 1:
        user_parts.append(
            f"Previous analogy used: {previous_analogy or 'unknown — pick a fresh one'}"
        )
        user_parts.append(f"This is retry attempt {retry_n}. Use a DIFFERENT analogy.")
    user_parts.append("Write the explanation now.")

    return ask_llm(
        system_prompt,
        "\n".join(user_parts),
        model=model,
        temperature=_EXPLANATION_TEMPERATURE,
    ).strip()


def check_understanding(
    concept_attrs: dict[str, Any],
    explanation: str,
    student_followup: str,
    *,
    model: str = DEFAULT_MODEL,
    max_json_retries: int = MAX_UNDERSTANDING_JSON_RETRIES,
) -> dict[str, Any]:
    """Classify whether the student understood the teaching explanation.

    Retries with :data:`UNDERSTANDING_CHECK_STRICT` on malformed JSON or
    invalid verdict labels (same pattern as Phase 3 classification).

    Args:
        concept_attrs: Graph node attributes for the taught concept.
        explanation: The tutor explanation that was just given.
        student_followup: The student's application response.
        model: Ollama model tag.
        max_json_retries: Additional JSON-parse retries after the first call.

    Returns:
        Dict with keys ``verdict``, ``confidence``, ``rationale``. On
        exhausted retries, returns ``still_struggling`` with confidence 0.0.

    Raises:
        OllamaError: Infrastructure failures propagate (not retried here).
    """
    user_message = (
        f"Concept: {concept_attrs.get('name', 'Unknown')}\n"
        f"Explanation given:\n{explanation}\n\n"
        f"Student follow-up:\n{student_followup}\n\n"
        "Classify understanding as JSON now."
    )

    last_error = "no attempts made"
    for attempt in range(max_json_retries + 1):
        system_prompt = (
            UNDERSTANDING_CHECK
            if attempt == 0
            else UNDERSTANDING_CHECK_STRICT
        )
        temperature = (
            _UNDERSTANDING_TEMPERATURE_FIRST
            if attempt == 0
            else _UNDERSTANDING_TEMPERATURE_RETRY
        )
        try:
            data = ask_llm_json(
                system_prompt,
                user_message,
                model=model,
                temperature=temperature,
            )
        except json.JSONDecodeError as exc:
            last_error = f"attempt {attempt}: invalid JSON ({exc.msg})"
            continue
        except OllamaError:
            raise

        parsed = _parse_understanding_verdict(data)
        if parsed is not None:
            return parsed
        last_error = f"attempt {attempt}: bad payload ({data!r})"

    return {
        "verdict": "still_struggling",
        "confidence": 0.0,
        "rationale": (
            f"Could not classify understanding after {max_json_retries + 1} "
            f"attempts. Last error: {last_error}."
        ),
    }


def teach_concept(
    graph: nx.DiGraph,
    profile: StudentProfile,
    concept_id: str,
    misconception_collection: Collection | None,
    *,
    student_response_provider: Callable[[str], str],
    max_retries: int = MAX_TEACHING_RETRIES,
    model: str = DEFAULT_MODEL,
) -> TeachResult:
    """Teach one concept with explanation, follow-up, and limited retries.

    Args:
        graph: Concept graph.
        profile: Mutable student profile.
        concept_id: Concept to teach.
        misconception_collection: Optional ChromaDB collection for correction
            lookup when the profile stores a ``misconception_summary``.
        student_response_provider: Callable receiving the explanation text
            and returning the student's follow-up (used by Streamlit, CLI,
            or tests).
        max_retries: Maximum retries after the first attempt (default 2 =>
            3 total attempts).
        model: Ollama model tag.

    Returns:
        :class:`TeachResult` summarizing success, retries, and transcript.

    Raises:
        KeyError: If ``concept_id`` is not in ``graph``.
        OllamaError: On LLM infrastructure failure.
    """
    if concept_id not in graph:
        raise KeyError(f"Unknown concept id: {concept_id!r}")

    concept_attrs = dict(graph.nodes[concept_id])
    known_names = _known_concept_names(graph, profile)
    misconception_correction = _lookup_correction(
        profile, concept_id, misconception_collection
    )

    transcript: list[dict[str, Any]] = []
    previous_analogy: str | None = None
    last_explanation = ""

    for attempt in range(max_retries + 1):
        explanation = generate_explanation(
            concept_attrs,
            known_names,
            misconception_correction,
            retry_n=attempt,
            previous_analogy=previous_analogy,
            model=model,
        )
        last_explanation = explanation
        student_followup = student_response_provider(explanation)
        verdict = check_understanding(
            concept_attrs,
            explanation,
            student_followup,
            model=model,
        )

        transcript.append(
            {
                "attempt": attempt,
                "explanation": explanation,
                "student_followup": student_followup,
                "verdict": verdict,
            }
        )
        _append_teaching_history(
            profile,
            concept_id,
            explanation,
            student_followup,
            verdict,
        )

        if verdict.get("verdict") == "understood":
            _mark_concept_understood(
                profile,
                graph,
                concept_id,
                confidence=float(verdict.get("confidence", 0.8)),
                rationale=str(verdict.get("rationale", "Demonstrated understanding after teaching.")),
            )
            return TeachResult(
                concept_id=concept_id,
                success=True,
                retries_used=attempt,
                flagged_for_review=False,
                transcript=transcript,
            )

        previous_analogy = _guess_analogy_from_explanation(explanation)

    mark_for_review(
        profile,
        concept_id,
        reason=(
            f"Teaching exhausted after {max_retries + 1} attempts for "
            f"{concept_id!r}. Last explanation snippet: "
            f"{last_explanation[:120]}..."
        ),
    )
    return TeachResult(
        concept_id=concept_id,
        success=False,
        retries_used=max_retries,
        flagged_for_review=True,
        transcript=transcript,
    )


def _lookup_correction(
    profile: StudentProfile,
    concept_id: str,
    collection: Collection | None,
) -> str | None:
    """Retrieve a misconception correction from ChromaDB if available."""
    record = profile.concept_statuses.get(concept_id, {})
    summary = record.get("misconception_summary")
    if not summary or collection is None:
        return None
    match = find_correction(collection, str(summary))
    if match is None:
        return None
    return str(match.get("correction", "")) or None


def _parse_understanding_verdict(data: Any) -> dict[str, Any] | None:
    """Validate LLM JSON into a normalized understanding verdict dict."""
    if not isinstance(data, dict):
        return None
    verdict = data.get("verdict")
    if verdict not in _VALID_VERDICTS:
        return None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    rationale = str(data.get("rationale", "")).strip() or "no rationale given"
    return {
        "verdict": verdict,
        "confidence": confidence,
        "rationale": rationale,
    }


def _append_teaching_history(
    profile: StudentProfile,
    concept_id: str,
    explanation: str,
    student_followup: str,
    verdict: dict[str, Any],
) -> None:
    """Append one teaching turn to the profile conversation history."""
    now = utc_now_iso()
    profile.conversation_history.extend(
        [
            {
                "role": "system",
                "kind": "explanation",
                "concept_id": concept_id,
                "content": explanation,
                "timestamp": now,
            },
            {
                "role": "student",
                "kind": "followup",
                "concept_id": concept_id,
                "content": student_followup,
                "timestamp": now,
            },
            {
                "role": "system",
                "kind": "understanding_check",
                "concept_id": concept_id,
                "content": str(verdict.get("verdict", "")),
                "result": verdict,
                "timestamp": now,
            },
        ]
    )
    profile.updated_at = now


def _mark_concept_understood(
    profile: StudentProfile,
    graph: nx.DiGraph,
    concept_id: str,
    *,
    confidence: float,
    rationale: str,
) -> None:
    """Mark a concept as confirmed_known after successful teaching."""
    now = utc_now_iso()
    profile.concept_statuses[concept_id] = {
        "status": "confirmed_known",
        "confidence": confidence,
        "rationale": rationale,
        "misconception_summary": None,
        "last_seen": now,
    }
    set_node_status(graph, concept_id, diagnostic_to_graph_status("confirmed_known"))
    profile.updated_at = now
