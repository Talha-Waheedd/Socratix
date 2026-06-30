"""Phase 3 diagnostic agent: Socratic questioning + response classification.

Two LLM-backed entry points:

- :func:`generate_question` -> one open-ended Socratic question per concept.
- :func:`classify_response` -> student free-text -> structured
  :class:`DiagnosticResult`.

The Streamlit driver (Phase 9) calls these one at a time so the UI can pause
between question and response without blocking. The CLI smoke test does the
same with hardcoded or stdin-provided responses.

Local-LLM tradeoff: the 8B model occasionally emits malformed JSON or an
out-of-vocabulary classification label. :func:`classify_response` retries up
to :data:`MAX_CLASSIFICATION_RETRIES` times with a stricter, example-free
prompt before falling back to a safe ``confirmed_unknown`` result with
confidence 0.0 (per the project spec: do not crash the loop; let Phase 4
flag the concept for review).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from socratix.llm import (
    DEFAULT_MODEL,
    OllamaError,
    ask_llm,
    ask_llm_json,
)
from socratix.prompts import (
    DIAGNOSTIC_CLASSIFICATION,
    DIAGNOSTIC_CLASSIFICATION_STRICT,
    SOCRATIC_QUESTION_GENERATION,
)

Classification = Literal[
    "confirmed_known", "confirmed_unknown", "misconception_detected"
]

_VALID_CLASSES: frozenset[str] = frozenset(
    {"confirmed_known", "confirmed_unknown", "misconception_detected"}
)

MAX_CLASSIFICATION_RETRIES: int = 2
"""Maximum number of retries after the first classification attempt.

Total LLM calls per classification = 1 + MAX_CLASSIFICATION_RETRIES = 3.
"""

_QUESTION_GEN_TEMPERATURE: float = 0.6
_CLASSIFY_TEMPERATURE_FIRST: float = 0.3
_CLASSIFY_TEMPERATURE_RETRY: float = 0.1


@dataclass(frozen=True)
class DiagnosticResult:
    """Structured outcome of classifying one student response.

    Attributes:
        concept_id: The concept this diagnostic turn was about.
        classification: One of ``confirmed_known``, ``confirmed_unknown``,
            ``misconception_detected``.
        confidence: Model-reported confidence in ``[0.0, 1.0]``. A value of
            ``0.0`` paired with a "could not classify" rationale signals
            that the retry path exhausted and Phase 4 should flag the
            concept for human review.
        rationale: One-sentence model rationale for the classification.
        misconception_summary: A one-sentence summary of the specific wrong
            belief, populated only when ``classification`` is
            ``misconception_detected``. Phase 6 (ChromaDB misconception
            retrieval) embeds this string to find a matching correction.
        question: The Socratic question that produced this response.
        student_response: The student's verbatim response.
    """

    concept_id: str
    classification: Classification
    confidence: float
    rationale: str
    misconception_summary: str | None = None
    question: str | None = None
    student_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict view (used by Phase 4 persistence)."""
        return asdict(self)


def generate_question(
    concept_id: str,
    concept_name: str,
    concept_description: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = _QUESTION_GEN_TEMPERATURE,
) -> str:
    """Generate a single open-ended Socratic question about a concept.

    The system prompt (see :data:`socratix.prompts.SOCRATIC_QUESTION_GENERATION`)
    forbids yes/no questions, factual-recall questions, and preambles. We
    still defensively strip surrounding quotes / whitespace because 8B models
    occasionally wrap output in quotes despite the rule.

    Args:
        concept_id: The graph node id, e.g. ``"for_loops"``.
        concept_name: Human-readable name, e.g. ``"For Loops"``.
        concept_description: One-line description from the concept graph.
        model: Override the Ollama model. Defaults to the wrapper's default.
        temperature: Sampling temperature; 0.6 gives some variation across
            sessions without going off-topic.

    Returns:
        The question as a single string with no surrounding whitespace.

    Raises:
        OllamaError: If the local LLM is unavailable. Connection errors are
            not retried at this layer - the caller should handle them.
    """
    user_message = (
        f"Concept id: {concept_id}\n"
        f"Concept name: {concept_name}\n"
        f"Concept description: {concept_description}\n\n"
        "Produce the single Socratic question now."
    )
    reply = ask_llm(
        SOCRATIC_QUESTION_GENERATION,
        user_message,
        model=model,
        temperature=temperature,
    )
    return _clean_question(reply)


def classify_response(
    concept_id: str,
    concept_name: str,
    question: str,
    student_response: str,
    *,
    model: str = DEFAULT_MODEL,
    max_retries: int = MAX_CLASSIFICATION_RETRIES,
) -> DiagnosticResult:
    """Classify a free-text response into a :class:`DiagnosticResult`.

    Attempts the rich classification prompt first. On JSON parse failure or
    an invalid classification label, retries up to ``max_retries`` times
    with :data:`socratix.prompts.DIAGNOSTIC_CLASSIFICATION_STRICT`. After
    the final retry, returns a conservative
    ``confirmed_unknown`` / confidence=0.0 result rather than raising, so
    the diagnostic loop can continue and Phase 4 can flag the concept.

    Connection-level failures (:class:`socratix.llm.OllamaError`) are not
    retried here - they propagate so the UI can show a clear infrastructure
    error rather than silently masking it.

    Args:
        concept_id: The concept graph node id.
        concept_name: Human-readable concept name (used inside the user
            message for additional grounding).
        question: The Socratic question that was just asked.
        student_response: The student's verbatim free-text reply.
        model: Override the Ollama model.
        max_retries: How many additional attempts after the first call.
            Defaults to :data:`MAX_CLASSIFICATION_RETRIES`.

    Returns:
        A :class:`DiagnosticResult`. The ``question`` and
        ``student_response`` fields are populated so callers do not need
        to thread them through separately.
    """
    user_message = (
        f"Concept: {concept_name} (id: {concept_id})\n"
        f"Question asked: {question}\n"
        f"Student response: {student_response}\n\n"
        "Classify this response now as a JSON object."
    )

    last_error: str = "no attempts made"
    for attempt in range(max_retries + 1):
        system_prompt = (
            DIAGNOSTIC_CLASSIFICATION
            if attempt == 0
            else DIAGNOSTIC_CLASSIFICATION_STRICT
        )
        temperature = (
            _CLASSIFY_TEMPERATURE_FIRST
            if attempt == 0
            else _CLASSIFY_TEMPERATURE_RETRY
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
            # Infrastructure failure - do not silently retry; let it surface
            # so the UI shows a real error instead of misleading the user.
            raise

        parsed = _parse_classification(
            concept_id=concept_id,
            data=data,
            question=question,
            student_response=student_response,
        )
        if parsed is not None:
            return parsed

        last_error = f"attempt {attempt}: bad payload shape ({_short_repr(data)})"

    # All retries exhausted. Fall back rather than crash (per project spec).
    return DiagnosticResult(
        concept_id=concept_id,
        classification="confirmed_unknown",
        confidence=0.0,
        rationale=(
            f"Could not classify response after {max_retries + 1} attempts. "
            f"Last error: {last_error}."
        ),
        misconception_summary=None,
        question=question,
        student_response=student_response,
    )


def _parse_classification(
    *,
    concept_id: str,
    data: Any,
    question: str,
    student_response: str,
) -> DiagnosticResult | None:
    """Validate and coerce the LLM's JSON output into a DiagnosticResult.

    Returns None when the payload is malformed enough that a retry is
    warranted (caller will issue a stricter prompt). Returns a normalized
    DiagnosticResult on success.
    """
    if not isinstance(data, dict):
        return None

    classification = data.get("classification")
    if classification not in _VALID_CLASSES:
        return None

    confidence_raw = data.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(data.get("rationale", "")).strip() or "no rationale given"

    summary_raw = data.get("misconception_summary")
    misconception_summary: str | None
    if summary_raw is None:
        misconception_summary = None
    else:
        cleaned = str(summary_raw).strip()
        misconception_summary = cleaned or None

    # If model says "misconception_detected" but omitted the summary,
    # synthesize a minimal one from the rationale so Phase 6 still has
    # something to embed for retrieval.
    if classification == "misconception_detected" and not misconception_summary:
        misconception_summary = rationale

    return DiagnosticResult(
        concept_id=concept_id,
        classification=classification,  # type: ignore[arg-type]
        confidence=confidence,
        rationale=rationale,
        misconception_summary=misconception_summary,
        question=question,
        student_response=student_response,
    )


def _clean_question(reply: str) -> str:
    """Best-effort cleanup of an LLM-generated question string.

    Strips whitespace, removes a single layer of surrounding quotes, and
    drops common preamble labels like ``Question:`` even though the system
    prompt forbids them - 8B models slip these in occasionally.
    """
    text = reply.strip()
    for prefix in ("Question:", "Q:", "question:", "Socratic question:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return text


def _short_repr(obj: Any, limit: int = 160) -> str:
    """Compact repr for error messages, truncated to ``limit`` characters."""
    text = repr(obj)
    return text if len(text) <= limit else text[: limit - 3] + "..."
