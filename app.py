"""Socratix Streamlit frontend — diagnostic dialogue, teaching, live knowledge graph.

Run from the repository root:

    streamlit run app.py

Prerequisites: Ollama running with the default model pulled (see Phase 2).
First launch also builds the local ChromaDB misconception index if missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components

from socratix.concept_graph import get_graph_stats, load_and_build, load_concepts
from socratix.diagnostic import classify_response, generate_question
from socratix.gap_analyzer import find_gap_path, summarize_gap
from socratix.llm import OllamaError, is_ollama_running
from socratix.misconceptions import (
    DEFAULT_MISCONCEPTIONS_PATH,
    DEFAULT_PERSIST_DIR,
    build_misconception_db,
    get_collection,
)
from socratix.persistence import (
    DEFAULT_DB_PATH,
    db_to_student_profile,
    init_db,
    start_session,
    student_profile_to_db,
)
from socratix.student_model import (
    StudentProfile,
    apply_diagnostic_result,
    create_profile,
    mark_for_review,
    sync_graph_from_profile,
    utc_now_iso,
)
from socratix.teaching_agent import (
    MAX_TEACHING_RETRIES,
    check_understanding,
    generate_explanation,
    _mark_concept_understood,
)
from socratix.visualize import STATUS_COLORS, visualize_graph

REPO_ROOT = Path(__file__).resolve().parent
GRAPH_HTML = REPO_ROOT / "output" / "concept_graph.html"


# ---------------------------------------------------------------------------
# Cached resources (expensive to reload every Streamlit rerun)
# ---------------------------------------------------------------------------


@st.cache_resource
def cached_base_graph() -> nx.DiGraph:
    """Load the validated concept graph once per app process."""
    return load_and_build()


@st.cache_resource
def cached_misconception_collection():
    """Open or seed the local ChromaDB misconception collection."""
    if not DEFAULT_PERSIST_DIR.exists() or not any(DEFAULT_PERSIST_DIR.iterdir()):
        return build_misconception_db(DEFAULT_MISCONCEPTIONS_PATH)
    try:
        return get_collection()
    except Exception:
        return build_misconception_db(DEFAULT_MISCONCEPTIONS_PATH)


@st.cache_resource
def cached_concept_choices() -> list[tuple[str, str]]:
    """Return (id, display name) pairs for the target-topic picker."""
    data = load_concepts()
    return [(c["id"], c["name"]) for c in data["concepts"]]


def get_db_connection():
    """Return a SQLite connection for this Streamlit browser session.

    Stored in ``st.session_state`` (not ``@st.cache_resource``) because
    Streamlit reruns can execute on different threads; a cached connection
    from another thread triggers ``sqlite3.ProgrammingError``.
    """
    if "db_conn" not in st.session_state:
        st.session_state.db_conn = init_db(DEFAULT_DB_PATH)
    return st.session_state.db_conn


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "session_active": False,
        "profile": None,
        "graph": None,
        "session_id": None,
        "mode": "idle",
        "current_concept": None,
        "current_question": None,
        "current_explanation": None,
        "gap_path": [],
        "teaching_retry_n": 0,
        "previous_analogy": None,
        "awaiting_followup": False,
        "status_message": "Start a session from the sidebar to begin.",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _concept_name(graph: nx.DiGraph, concept_id: str | None) -> str:
    if not concept_id or concept_id not in graph:
        return concept_id or "Unknown"
    return str(graph.nodes[concept_id].get("name", concept_id))


def _known_concept_names(graph: nx.DiGraph, profile: StudentProfile) -> list[str]:
    names: list[str] = []
    for cid, record in profile.concept_statuses.items():
        if record.get("status") == "confirmed_known" and cid in graph:
            names.append(_concept_name(graph, cid))
    return names or ["Variables and Assignment", "Running Python"]


def _misconception_correction(profile: StudentProfile, concept_id: str) -> str | None:
    record = profile.concept_statuses.get(concept_id, {})
    summary = record.get("misconception_summary")
    if not summary:
        return None
    collection = cached_misconception_collection()
    from socratix.misconceptions import find_correction

    match = find_correction(collection, str(summary))
    return str(match["correction"]) if match else None


def _persist_profile() -> None:
    if st.session_state.profile and st.session_state.session_id:
        conn = get_db_connection()
        student_profile_to_db(
            conn,
            st.session_state.profile,
            st.session_state.session_id,
        )


def _append_teaching_entries(
    profile: StudentProfile,
    concept_id: str,
    explanation: str,
    followup: str | None,
    verdict: dict[str, Any] | None,
) -> None:
    now = utc_now_iso()
    profile.conversation_history.append(
        {
            "role": "system",
            "kind": "explanation",
            "concept_id": concept_id,
            "content": explanation,
            "timestamp": now,
        }
    )
    if followup is not None:
        profile.conversation_history.append(
            {
                "role": "student",
                "kind": "followup",
                "concept_id": concept_id,
                "content": followup,
                "timestamp": now,
            }
        )
    if verdict is not None:
        profile.conversation_history.append(
            {
                "role": "system",
                "kind": "understanding_check",
                "concept_id": concept_id,
                "content": str(verdict.get("verdict", "")),
                "result": verdict,
                "timestamp": now,
            }
        )
    profile.updated_at = now


def _advance_gap_path() -> None:
    gap: list[str] = st.session_state.gap_path
    current = st.session_state.current_concept
    if gap and gap[0] == current:
        gap.pop(0)
    elif current in gap:
        gap.remove(current)
    st.session_state.gap_path = gap


def _begin_diagnose_concept(concept_id: str) -> None:
    graph: nx.DiGraph = st.session_state.graph
    profile: StudentProfile = st.session_state.profile
    attrs = graph.nodes[concept_id]

    st.session_state.mode = "diagnose"
    st.session_state.current_concept = concept_id
    st.session_state.current_explanation = None
    st.session_state.awaiting_followup = False
    st.session_state.teaching_retry_n = 0
    st.session_state.previous_analogy = None

    with st.spinner(f"Preparing a question about {_concept_name(graph, concept_id)}..."):
        question = generate_question(concept_id, attrs["name"], attrs["description"])
    st.session_state.current_question = question
    st.session_state.status_message = (
        f"Diagnosing: {_concept_name(graph, concept_id)}"
    )


def _begin_teach_concept(concept_id: str, *, retry_n: int = 0) -> None:
    graph: nx.DiGraph = st.session_state.graph
    profile: StudentProfile = st.session_state.profile
    attrs = graph.nodes[concept_id]

    st.session_state.mode = "teach"
    st.session_state.current_concept = concept_id
    st.session_state.current_question = None
    st.session_state.teaching_retry_n = retry_n
    st.session_state.awaiting_followup = True

    correction = _misconception_correction(profile, concept_id)
    label = "Retrying with a different explanation" if retry_n else "Preparing a lesson"
    with st.spinner(f"{label} for {_concept_name(graph, concept_id)}..."):
        explanation = generate_explanation(
            attrs,
            _known_concept_names(graph, profile),
            correction,
            retry_n=retry_n,
            previous_analogy=st.session_state.previous_analogy,
        )
    st.session_state.current_explanation = explanation
    st.session_state.status_message = (
        f"Teaching: {_concept_name(graph, concept_id)} "
        f"(attempt {retry_n + 1} of {MAX_TEACHING_RETRIES + 1})"
    )


def _setup_next_step() -> None:
    gap: list[str] = st.session_state.gap_path
    if not gap:
        st.session_state.mode = "idle"
        st.session_state.current_concept = None
        st.session_state.current_question = None
        st.session_state.current_explanation = None
        st.session_state.awaiting_followup = False
        st.session_state.status_message = (
            "Session complete — you have covered all concepts on the learning path."
        )
        return
    _begin_diagnose_concept(gap[0])


def _handle_diagnose_input(user_text: str) -> None:
    graph: nx.DiGraph = st.session_state.graph
    profile: StudentProfile = st.session_state.profile
    concept_id: str = st.session_state.current_concept
    question: str = st.session_state.current_question
    attrs = graph.nodes[concept_id]

    result = classify_response(concept_id, attrs["name"], question, user_text)
    apply_diagnostic_result(profile, graph, result)
    _persist_profile()

    st.session_state.current_question = None

    if result.classification == "confirmed_known":
        _advance_gap_path()
        _setup_next_step()
    else:
        _begin_teach_concept(concept_id, retry_n=0)


def _handle_teach_input(user_text: str) -> None:
    graph: nx.DiGraph = st.session_state.graph
    profile: StudentProfile = st.session_state.profile
    concept_id: str = st.session_state.current_concept
    explanation: str = st.session_state.current_explanation
    attrs = graph.nodes[concept_id]

    verdict = check_understanding(attrs, explanation, user_text)
    _append_teaching_entries(profile, concept_id, explanation, user_text, verdict)
    _persist_profile()

    st.session_state.awaiting_followup = False

    if verdict.get("verdict") == "understood":
        _mark_concept_understood(
            profile,
            graph,
            concept_id,
            confidence=float(verdict.get("confidence", 0.8)),
            rationale=str(verdict.get("rationale", "Understood after teaching.")),
        )
        _persist_profile()
        st.session_state.current_explanation = None
        _advance_gap_path()
        _setup_next_step()
        return

    retry_n = st.session_state.teaching_retry_n + 1
    if retry_n <= MAX_TEACHING_RETRIES:
        from socratix.teaching_agent import _guess_analogy_from_explanation

        st.session_state.previous_analogy = _guess_analogy_from_explanation(explanation)
        _begin_teach_concept(concept_id, retry_n=retry_n)
        return

    mark_for_review(
        profile,
        concept_id,
        reason="Teaching retries exhausted in Streamlit session.",
    )
    _persist_profile()
    st.session_state.current_explanation = None
    _advance_gap_path()
    _setup_next_step()


def _start_session(student_id: str, target_concept: str) -> None:
    conn = get_db_connection()
    row = conn.execute(
        "SELECT 1 FROM students WHERE id = ?",
        (student_id,),
    ).fetchone()

    if row:
        profile = db_to_student_profile(
            conn,
            student_id,
            target_concept,
            include_conversation=True,
        )
    else:
        profile = create_profile(student_id, target_concept)

    session_id = start_session(conn, student_id, target_concept)
    graph = cached_base_graph().copy()
    sync_graph_from_profile(profile, graph)
    gap_path = find_gap_path(graph, profile, target_concept)

    st.session_state.session_active = True
    st.session_state.profile = profile
    st.session_state.graph = graph
    st.session_state.session_id = session_id
    st.session_state.gap_path = gap_path
    st.session_state.previous_analogy = None
    st.session_state.teaching_retry_n = 0
    st.session_state.awaiting_followup = False

    if not gap_path:
        st.session_state.mode = "idle"
        st.session_state.status_message = (
            f"You already know everything needed for "
            f"{_concept_name(graph, target_concept)}."
        )
        student_profile_to_db(conn, profile, session_id)
        return

    _begin_diagnose_concept(gap_path[0])


def _render_sidebar() -> tuple[str, str]:
    st.sidebar.title("Socratix")
    st.sidebar.caption("Adaptive Python tutor — local LLM, personal knowledge graph.")

    if not is_ollama_running():
        st.sidebar.error(
            "Ollama is not running. Start it with `ollama serve`, then refresh."
        )

    concept_choices = cached_concept_choices()
    id_to_name = {cid: name for cid, name in concept_choices}
    name_to_id = {name: cid for cid, name in concept_choices}

    student_id = st.sidebar.text_input("Student ID", value="student")
    target_name = st.sidebar.selectbox(
        "Target topic",
        options=[name for _, name in concept_choices],
        index=next(
            (i for i, (cid, _) in enumerate(concept_choices) if cid == "recursion_basics"),
            0,
        ),
    )
    target_concept = name_to_id[target_name]

    if st.sidebar.button("Start session", type="primary", disabled=not is_ollama_running()):
        try:
            _start_session(student_id.strip() or "student", target_concept)
        except OllamaError as exc:
            st.sidebar.error(str(exc))

    if st.session_state.session_active and st.session_state.graph is not None:
        graph: nx.DiGraph = st.session_state.graph
        stats = get_graph_stats(graph)
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Knowledge graph**")
        for label, color in [
            ("Known", STATUS_COLORS["known"]),
            ("Gap / unknown", STATUS_COLORS["unknown"]),
            ("Misconception", STATUS_COLORS["misconception"]),
            ("Not assessed", STATUS_COLORS["unassessed"]),
        ]:
            st.sidebar.markdown(
                f'<span style="color:{color}">&#9679;</span> {label}',
                unsafe_allow_html=True,
            )

        status_counts = stats.get("statuses", {})
        st.sidebar.markdown(
            f"Known **{status_counts.get('known', 0)}** · "
            f"Unknown **{status_counts.get('unknown', 0)}** · "
            f"Misconception **{status_counts.get('misconception', 0)}** · "
            f"Unassessed **{status_counts.get('unassessed', 0)}**"
        )

        html_path = visualize_graph(graph, GRAPH_HTML, height="500px")
        components.html(
            html_path.read_text(encoding="utf-8"),
            height=520,
            scrolling=True,
        )

        if st.session_state.gap_path:
            st.sidebar.markdown("**Learning path**")
            st.sidebar.caption(
                summarize_gap(graph, st.session_state.gap_path, target_concept_id=target_concept)
            )

        profile: StudentProfile | None = st.session_state.profile
        if profile and profile.needs_review:
            with st.sidebar.expander("Needs human review"):
                for cid in profile.needs_review:
                    st.write(f"- {_concept_name(graph, cid)} (`{cid}`)")

    return student_id, target_concept


def _render_chat() -> None:
    st.title("Socratix Tutor")
    st.info(st.session_state.status_message)

    profile: StudentProfile | None = st.session_state.profile
    if profile:
        for entry in profile.conversation_history:
            role = entry.get("role", "system")
            content = entry.get("content", "")
            kind = entry.get("kind", "")
            prefix = f"*{kind}* — " if kind else ""
            if role == "student":
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant"):
                    if kind in {"explanation", "question"}:
                        st.markdown(f"{prefix}{content}")
                    elif kind == "classification":
                        st.markdown(
                            f"*Classification:* **{content}** — "
                            f"{entry.get('result', {}).get('rationale', '')}"
                        )
                    elif kind == "understanding_check":
                        st.markdown(
                            f"*Understanding check:* **{content}** — "
                            f"{entry.get('result', {}).get('rationale', '')}"
                        )
                    else:
                        st.markdown(f"{prefix}{content}")

    if st.session_state.mode == "diagnose" and st.session_state.current_question:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.current_question)

    if st.session_state.mode == "teach" and st.session_state.current_explanation:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.current_explanation)

    if (
        st.session_state.session_active
        and st.session_state.mode in {"diagnose", "teach"}
        and is_ollama_running()
    ):
        prompt = (
            "Apply the concept in your own words..."
            if st.session_state.mode == "teach"
            else "Your answer..."
        )
        user_input = st.chat_input(prompt)
        if user_input:
            try:
                if st.session_state.mode == "diagnose":
                    _handle_diagnose_input(user_input)
                elif st.session_state.mode == "teach" and st.session_state.awaiting_followup:
                    _handle_teach_input(user_input)
            except OllamaError as exc:
                st.error(str(exc))
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Socratix",
        page_icon=":material/school:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
