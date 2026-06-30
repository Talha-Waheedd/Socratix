# Socratix - Complete Build Plan and Handoff

This document is a self-contained handoff for the Socratix project. Three of
the ten phases are implemented and verified; seven remain. Read this
front-to-back before continuing. Every remaining phase has: goal, files to
create, key function signatures, schemas / prompts / data, a concrete
algorithm where relevant, edge cases, and a runnable test.

If you resume with a less-capable model, give it ONE phase at a time, not
the whole document. The "How to resume with a free model" section at the
bottom explains the workflow.

---

## 1. Project overview

Socratix is an adaptive AI tutoring agent that:

1. Diagnoses what a student knows through Socratic dialogue.
2. Builds a personal knowledge graph of their understanding.
3. Teaches only the specific concepts they are missing, using analogies
   drawn from concepts they already know.

Domain (V1): Python programming fundamentals through basic data
structures, simple algorithms, and an intro to time complexity. The concept
graph has 50 nodes across 12 categories.

Hard constraint: 100 percent free, local, open-source. No paid APIs of
any kind.

---

## 2. Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| LLM | Ollama running `llama3.1:8b` (fallback `mistral:7b`) | Fully local, REST API at `http://localhost:11434` |
| Knowledge graph | NetworkX | Pure Python, no external service |
| Vector store | ChromaDB (local persistent client) | No cloud account, no API key |
| Embeddings | sentence-transformers, model `all-MiniLM-L6-v2` | ~80 MB, runs on CPU |
| Progress storage | SQLite | Built into Python standard library |
| Frontend | Streamlit | Free, runs locally |
| Graph visualization | Pyvis | Interactive HTML output |

Final `requirements.txt` after Phase 10:

```
networkx==3.3
pyvis==0.3.2
requests==2.32.3
chromadb==0.5.5
sentence-transformers==3.0.1
streamlit==1.39.0
```

---

## 3. Working rules (apply to every phase)

1. After every phase, a runnable test under `scripts/test_phaseN.py` must
   pass before moving on.
2. All LLM system prompts live in `socratix/prompts.py`. Never inline a
   prompt in business-logic code.
3. Type hints and docstrings on every public function.
4. If the 8B model produces inconsistent JSON, retry with a stricter
   prompt. The pattern is implemented in Phase 3 (`classify_response` +
   `DIAGNOSTIC_CLASSIFICATION_STRICT`); reuse it for every other JSON
   classification (Phase 7 understanding check).
5. Flag local-LLM tradeoffs in comments wherever the 8B model's smaller
   size will materially reduce quality compared to a frontier model.
6. Use UTC ISO 8601 for all timestamps:
   `datetime.now(timezone.utc).isoformat()`.

---

## 4. Current status

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Concept graph foundation | COMPLETED |
| 2 | Local LLM connection | COMPLETED |
| 3 | Diagnostic agent | COMPLETED |
| 4 | Student model | REMAINING |
| 5 | Gap analyzer | REMAINING |
| 6 | Misconception database | REMAINING |
| 7 | Teaching agent | REMAINING |
| 8 | Progress persistence | REMAINING |
| 9 | Streamlit frontend | REMAINING |
| 10 | Polish and README | REMAINING |

---

## PHASE 1 - Concept graph foundation - COMPLETED

Goal: an editable NetworkX DAG of 40-60 Python concepts with prerequisite
edges, plus a Pyvis HTML visualization.

Files in the repo:

- `data/concepts.json` - 50 concepts in 12 categories with ~74
  prerequisite edges. Top-level keys: `version`, `domain`, `description`,
  `concepts[]`. Each concept has `id`, `name`, `description`, `category`,
  `prerequisites: list[str]`.
- `socratix/concept_graph.py`:
  - `load_concepts(path) -> dict`
  - `build_graph(concepts_data) -> nx.DiGraph` (node attrs: `name`,
    `description`, `category`, `prerequisites`, `status="unassessed"`)
  - `validate_graph(graph) -> None` (raises on dangling prereq, cycles,
    or node count outside `[MIN_NODES=40, MAX_NODES=60]`)
  - `set_node_status(graph, concept_id, status)` where status is one of
    `known | unknown | misconception | unassessed`
  - `get_graph_stats(graph) -> dict`
  - `load_and_build(path=DEFAULT_CONCEPTS_PATH) -> nx.DiGraph` convenience
- `socratix/visualize.py`:
  - `visualize_graph(graph, output_path, height="750px", width="100%",
    open_browser=False) -> Path`
  - Status colors (hex): known=`#22c55e` green, unknown=`#ef4444` red,
    misconception=`#f97316` orange, unassessed=`#9ca3af` gray.
  - Layout: `barnes_hut` physics, `cdn_resources="remote"`, `notebook=False`.
- `scripts/test_phase1.py` - asserts 40-60 nodes, DAG, writes Pyvis HTML
  with three sample status assignments for color verification.

Verify: `python scripts/test_phase1.py`. Open
`output/concept_graph.html` in a browser.

---

## PHASE 2 - Local LLM connection - COMPLETED

Goal: a typed wrapper over the Ollama REST API.

Files in the repo:

- `socratix/llm.py`:
  - Constants: `OLLAMA_BASE_URL` (env `OLLAMA_BASE_URL`),
    `DEFAULT_MODEL` (env `OLLAMA_MODEL`, default `llama3.1:8b`),
    `DEFAULT_TIMEOUT_SECONDS=120.0`, `DEFAULT_TEMPERATURE=0.4`.
  - Exceptions: `OllamaError` (base), `OllamaUnavailableError` (cannot
    reach), `OllamaModelMissingError` (not pulled),
    `OllamaResponseError` (other non-2xx / bad body).
  - `is_ollama_running(base_url) -> bool` (never raises).
  - `list_local_models(base_url) -> list[str]`.
  - `ask_llm(system_prompt, user_message, *, model, temperature, json_mode,
    history, timeout, base_url) -> str` - primary entry point.
  - `ask_llm_json(...)` - calls `ask_llm` with `json_mode=True` and
    `json.loads` the result. Does NOT retry on parse failure (callers
    handle retry, e.g. `diagnostic.classify_response`).
- `scripts/test_phase2.py` - 3 checks: Ollama reachable, model present,
  real `ask_llm` call. Exits with clear actionable hints on failure.

Verify (after installing Ollama and pulling the model):

```
ollama pull llama3.1:8b
python scripts/test_phase2.py
```

Tradeoff note in `llm.py`: 8B models are weaker at strict JSON formatting
and multi-step reasoning. Set `OLLAMA_MODEL=mistral:7b` if 8B is too slow.

---

## PHASE 3 - Diagnostic agent (Socratic questioning) - COMPLETED

Goal: ask ONE open-ended Socratic question per concept, classify the
student's free-text response into one of three categories.

Files in the repo:

- `socratix/prompts.py`:
  - `SOCRATIC_QUESTION_GENERATION` - forbids yes/no, factual recall,
    preambles. Gives 3 good and 4 bad examples.
  - `DIAGNOSTIC_CLASSIFICATION` - rich prompt with 5 worked examples
    spanning all 3 categories.
  - `DIAGNOSTIC_CLASSIFICATION_STRICT` - bare schema, no examples. Used
    on retry only.
- `socratix/diagnostic.py`:
  - `DiagnosticResult` dataclass: `concept_id`, `classification` (literal
    type for the 3 labels), `confidence` float in `[0,1]`, `rationale`,
    `misconception_summary` (str or None, synthesized from rationale
    when missing), `question`, `student_response`. `to_dict()` returns a
    JSON-serializable dict.
  - `generate_question(concept_id, concept_name, concept_description, *,
    model, temperature=0.6) -> str`. Cleans output with
    `_clean_question` (strips wrapping quotes, "Question:" / "Q:" /
    "Socratic question:" prefixes).
  - `classify_response(concept_id, concept_name, question,
    student_response, *, model, max_retries=2) -> DiagnosticResult`.
    Retries with the strict prompt on `json.JSONDecodeError` or bad
    payload shape. Final fallback (after retries exhausted) returns
    `confirmed_unknown` with `confidence=0.0` and a "Could not classify"
    rationale. Does NOT retry on `OllamaError` (infrastructure
    failures bubble up).
- `scripts/test_phase3.py` - 3 canned scenarios (one expected per
  category) plus `--interactive` mode with optional `--concept` flag.

Verify (after Phase 2 passes): `python scripts/test_phase3.py`

Pass bar: 2/3 or 3/3 classifications match expected. If 0-1/3, tune
prompts in `socratix/prompts.py` before moving on.

Pattern to reuse: every later JSON classification (Phase 7 understanding
check) should copy this retry-with-stricter-prompt structure.

---

## 5. Repository layout

After Phase 3 (current):

```
d:\Socratix\
|-- data\
|   `-- concepts.json
|-- socratix\
|   |-- __init__.py
|   |-- concept_graph.py
|   |-- visualize.py
|   |-- llm.py
|   |-- prompts.py
|   `-- diagnostic.py
|-- scripts\
|   |-- test_phase1.py
|   |-- test_phase2.py
|   `-- test_phase3.py
|-- output\
|   `-- concept_graph.html        (generated)
|-- requirements.txt
|-- .gitignore
`-- PROJECT_PLAN.md               (this file)
```

After Phase 10 (final):

```
d:\Socratix\
|-- data\
|   |-- concepts.json
|   `-- misconceptions.json
|-- socratix\
|   |-- __init__.py
|   |-- concept_graph.py
|   |-- visualize.py
|   |-- llm.py
|   |-- prompts.py
|   |-- diagnostic.py
|   |-- student_model.py          (Phase 4)
|   |-- gap_analyzer.py           (Phase 5)
|   |-- misconceptions.py         (Phase 6)
|   |-- teaching_agent.py         (Phase 7)
|   `-- persistence.py            (Phase 8)
|-- scripts\
|   |-- test_phase1.py
|   |-- test_phase2.py
|   |-- test_phase3.py
|   |-- test_phase4.py
|   |-- test_phase5.py
|   |-- test_phase6.py
|   |-- test_phase7.py
|   `-- test_phase8.py
|-- app.py                        (Phase 9, Streamlit entry point)
|-- student_profiles\             (gitignored)
|-- chroma_db\                    (gitignored)
|-- socratix.sqlite               (gitignored)
|-- output\                       (gitignored)
|-- README.md                     (Phase 10)
|-- requirements.txt
|-- .gitignore
`-- PROJECT_PLAN.md
```

---

## PHASE 4 - Student model - REMAINING

Goal: persist per-session state (status per concept + full conversation
history) as JSON. Sync the concept graph's `status` node attribute from
this state.

Files to create:

- `socratix/student_model.py`
- `scripts/test_phase4.py`
- `student_profiles/` directory (already gitignored)

JSON profile schema (saved as
`student_profiles/<student_id>.json`):

```
{
  "student_id": "test_student_001",
  "created_at": "2026-06-30T19:00:00+00:00",
  "updated_at": "2026-06-30T19:42:13+00:00",
  "target_concept": "recursion_basics",
  "concept_statuses": {
    "for_loops": {
      "status": "confirmed_known",
      "confidence": 0.9,
      "rationale": "Correctly explains use-case distinction.",
      "misconception_summary": null,
      "last_seen": "2026-06-30T19:10:00+00:00"
    },
    "base_case": {
      "status": "misconception_detected",
      "confidence": 0.85,
      "rationale": "Conflates base case with function entry.",
      "misconception_summary": "Believes the base case is the first line executed.",
      "last_seen": "2026-06-30T19:12:00+00:00"
    }
  },
  "needs_review": ["unclear_concept_id"],
  "conversation_history": [
    {"role": "system",  "kind": "question",       "concept_id": "for_loops", "content": "When would you ...", "timestamp": "..."},
    {"role": "student", "kind": "response",       "concept_id": "for_loops", "content": "When I want to ...", "timestamp": "..."},
    {"role": "system",  "kind": "classification", "concept_id": "for_loops", "content": "confirmed_known", "result": { ... full DiagnosticResult.to_dict() ... }, "timestamp": "..."}
  ]
}
```

Status mapping (diagnostic label -> profile/graph status):

| DiagnosticResult.classification | profile/graph status |
|---------------------------------|----------------------|
| `confirmed_known`               | `known`              |
| `confirmed_unknown`             | `unknown`            |
| `misconception_detected`        | `misconception`      |

The profile stores the verbose diagnostic label
(`confirmed_known`) under `concept_statuses[id].status`; the graph stores
the short label (`known`). Provide a helper to map between them.

Key functions:

```
def create_profile(student_id: str, target_concept: str) -> StudentProfile: ...

def load_profile(path: Path) -> StudentProfile: ...

def save_profile(profile: StudentProfile, path: Path) -> None: ...

def apply_diagnostic_result(
    profile: StudentProfile,
    graph: nx.DiGraph,
    result: DiagnosticResult,
) -> None:
    # 1. Compute graph_status = _to_graph_status(result.classification).
    # 2. Update profile.concept_statuses[result.concept_id] with the
    #    full diagnostic record + last_seen timestamp.
    # 3. set_node_status(graph, result.concept_id, graph_status).
    # 4. Append 3 entries to conversation_history: question, response,
    #    classification (use DiagnosticResult.to_dict() in the classification
    #    entry).
    # 5. If result.confidence == 0.0, also call mark_for_review.
    # 6. Update profile.updated_at.

def sync_graph_from_profile(profile: StudentProfile, graph: nx.DiGraph) -> None:
    # Use when loading an existing profile to restore graph state.
    # Iterate concept_statuses and call set_node_status accordingly.

def mark_for_review(profile: StudentProfile, concept_id: str, reason: str) -> None:
    # Append to profile.needs_review if not already present.
```

Edge cases:

- Fresh profile: every concept in the graph starts implicitly
  `unassessed`. Do NOT populate concept_statuses for every node; leave
  the map empty and rely on the graph default. This keeps profile files
  small.
- Loading: if a concept_id in the profile is no longer in the graph
  (someone edited concepts.json), log a warning and skip it. Do not crash.
- Concurrent saves: write to a temp file then rename, so a crash
  mid-write does not corrupt the profile.

Test plan (`scripts/test_phase4.py`):

1. Build the graph. Create a fresh profile for "test_student_001",
   target "recursion_basics".
2. Apply three synthetic `DiagnosticResult` objects (one of each
   classification) for three different concepts.
3. Save the profile to `student_profiles/test_student_001.json`.
4. Load it back.
5. Assert: graph node statuses match the profile, conversation_history
   has 9 entries (3 turns x 3 entries each), the loaded profile is
   value-equal to the saved one, `needs_review` is empty.
6. Apply a fourth result with `confidence=0.0`. Assert
   `needs_review` contains the corresponding concept_id.

---

## PHASE 5 - Gap analyzer - REMAINING

Goal: given a target concept and the current student profile, return the
minimum ordered list of concepts to teach (unknown prerequisites in
topological order).

Files to create:

- `socratix/gap_analyzer.py`
- `scripts/test_phase5.py`

Algorithm (document this in a module-level comment):

We use `nx.ancestors(target) | {target}` to get all transitive
prerequisites of the target, then filter to those NOT yet
`confirmed_known`, then return them in topological order using
`nx.topological_sort` on the induced subgraph.

Why this over `nx.shortest_path`:

- `shortest_path` returns ONE path between two specific nodes. We want
  every unknown prerequisite, possibly across multiple ancestor branches
  (e.g. `binary_search` depends on both `linear_search` and
  `recursion_basics`).
- Topological sort on the induced subgraph gives the correct teaching
  order automatically: a concept always appears after its prerequisites.

Pseudocode:

```
def find_gap_path(
    graph: nx.DiGraph,
    profile: StudentProfile,
    target_concept_id: str,
) -> list[str]:
    if target_concept_id not in graph:
        raise KeyError(f"Unknown concept: {target_concept_id}")

    candidates = nx.ancestors(graph, target_concept_id) | {target_concept_id}

    def needs_teaching(cid: str) -> bool:
        record = profile.concept_statuses.get(cid)
        if record is None:
            return True  # unassessed
        return record.get("status") != "confirmed_known"

    gap = {c for c in candidates if needs_teaching(c)}
    subgraph = graph.subgraph(gap)
    return list(nx.topological_sort(subgraph))
```

Additional helper:

```
def summarize_gap(graph: nx.DiGraph, gap_path: list[str]) -> str:
    # Returns a multi-line human-readable summary for the Streamlit
    # sidebar, e.g.:
    #   "To learn 'Recursion Basics' you need to cover 4 concepts in
    #    this order: function_basics, return_values, ..."
```

Test plan:

1. Build graph + empty profile.
2. Mark `running_python`, `variables`, `data_types`, `operators`,
   `boolean_logic`, `if_statements`, `function_basics`, `return_values`
   as `confirmed_known` in the profile (via direct
   `concept_statuses` manipulation).
3. Call `find_gap_path(target="recursion_basics")`.
4. Assert: returned list includes `recursion_basics` AND any unknown
   prerequisites; does NOT include any of the 8 known concepts;
   topological order holds (every concept appears after all its
   prerequisites in the list).
5. Mark more concepts as known, re-run, assert the gap shrinks
   monotonically.
6. Edge case: if the target is already known, `find_gap_path` returns
   an empty list. Test this.
7. Edge case: target not in graph -> raises `KeyError`. Test this.

---

## PHASE 6 - Misconception database - REMAINING

Goal: a ChromaDB local persistent collection seeded with 20 common
Python misconceptions, each paired with a targeted correction. Given a
detected misconception (text), retrieve the closest matching correction.

Files to create:

- `data/misconceptions.json` - seed dataset (provided below).
- `socratix/misconceptions.py`
- `scripts/test_phase6.py`
- `chroma_db/` directory (already gitignored).

requirements.txt additions:

```
chromadb==0.5.5
sentence-transformers==3.0.1
```

Seed dataset (paste this exact JSON into `data/misconceptions.json`):

```
{
  "version": "1.0",
  "misconceptions": [
    {"id": "recursion_loop_inside", "concept_id": "recursion_basics", "misconception": "Recursion needs a loop inside it to make things repeat.", "correction": "Recursion repeats by having the function call ITSELF on a smaller version of the problem. The recursive call is what creates the repetition - no loop is needed inside the function body."},
    {"id": "base_case_first_line", "concept_id": "base_case", "misconception": "The base case is the first line of a recursive function and always runs before anything else.", "correction": "The base case is the TERMINATING condition that stops recursion and returns directly. It runs only when the input is small enough to answer without recursing further; on larger inputs the recursive case runs first."},
    {"id": "variables_old_value_attr", "concept_id": "variables", "misconception": "When you reassign a variable, Python keeps the old value accessible as an attribute.", "correction": "When you reassign a variable in Python, the name is simply rebound to the new value. The old value is no longer accessible through that name and is garbage-collected if no other reference exists."},
    {"id": "variable_is_a_box", "concept_id": "variables", "misconception": "A Python variable is a box that holds a value.", "correction": "A Python variable is a NAME that REFERS to an object in memory, not a container that holds it. Multiple variables can refer to the same object (a = [1]; b = a)."},
    {"id": "for_loop_only_lists", "concept_id": "for_loops", "misconception": "for loops can only iterate over lists.", "correction": "for loops iterate over ANY iterable: lists, tuples, strings, dicts, sets, ranges, generators, file objects, and any object that implements __iter__."},
    {"id": "while_runs_once", "concept_id": "while_loops", "misconception": "A while loop always runs at least once.", "correction": "A while loop checks its condition BEFORE the first iteration. If the condition is False initially, the loop body never runs. You may be thinking of do-while loops in other languages; Python has no do-while."},
    {"id": "lists_fixed_size", "concept_id": "list_basics", "misconception": "Python lists have a fixed size like arrays in C or Java.", "correction": "Python lists are dynamic. You can append, insert, or remove elements at any time. There is no need to declare a size upfront."},
    {"id": "negative_index_error", "concept_id": "list_indexing_slicing", "misconception": "Negative indices throw an IndexError in Python.", "correction": "Negative indices count from the END of the sequence: list[-1] is the last element, list[-2] the second to last, and so on. They only raise IndexError if their absolute value exceeds the length."},
    {"id": "dict_keys_strings_only", "concept_id": "dict_basics", "misconception": "Dictionary keys must be strings.", "correction": "Dictionary keys can be any HASHABLE object: strings, numbers, tuples, frozensets, None, booleans, and more. They cannot be mutable types like lists, dicts, or sets."},
    {"id": "tuples_just_immutable_lists", "concept_id": "tuples", "misconception": "Tuples are just lists you can not change - there is no other meaningful difference.", "correction": "Beyond immutability, tuples are hashable (can be dict keys / set members), use less memory, are slightly faster, and signal intent: a tuple typically means 'a fixed-shape grouping' (like coordinates) while a list means 'a collection of similar items'."},
    {"id": "strings_mutable", "concept_id": "string_basics", "misconception": "You can change a character in a string by assigning to an index: s[0] = 'A'.", "correction": "Strings in Python are IMMUTABLE. To change a character you must build a new string, e.g. s = 'A' + s[1:]."},
    {"id": "function_must_return", "concept_id": "function_basics", "misconception": "Functions must return a value, otherwise they crash.", "correction": "Functions without a return statement (or with a bare 'return') implicitly return None. This is perfectly valid; many functions exist for side effects (print, list.append) and intentionally return None."},
    {"id": "code_after_return_runs", "concept_id": "return_values", "misconception": "Code after a return statement still runs.", "correction": "A return statement immediately exits the function. Any code after it inside that branch is unreachable. (try/finally is the only exception: finally blocks run even after a return.)"},
    {"id": "local_visible_everywhere", "concept_id": "scope", "misconception": "Variables declared inside a function are accessible anywhere in the program.", "correction": "Variables defined inside a function are LOCAL to that function. Outside code cannot see them. To share data, pass arguments in or return values out."},
    {"id": "lambda_multi_statement", "concept_id": "lambda_functions", "misconception": "A lambda can have multiple statements separated by semicolons.", "correction": "A lambda must contain a SINGLE expression. For multiple statements or any statement (if/for/print/etc), use a normal def function."},
    {"id": "linear_search_fastest", "concept_id": "linear_search", "misconception": "Linear search is the fastest way to find an element in a sequence.", "correction": "Linear search is O(n) and only optimal for unsorted data or very small inputs. If the data is sorted, binary search at O(log n) is dramatically faster on large inputs."},
    {"id": "binary_search_unsorted", "concept_id": "binary_search", "misconception": "Binary search works on any list, regardless of order.", "correction": "Binary search requires the input to be SORTED. On unsorted input it may miss the element entirely. If you need fast lookup on unsorted data, build a set or dict first."},
    {"id": "on_always_faster", "concept_id": "big_o_intro", "misconception": "An O(n) algorithm is always faster than an O(n^2) algorithm.", "correction": "Big O ignores constants. For small inputs an O(n^2) algorithm with low constants can be faster than an O(n) one with high constants. Big O describes growth as n becomes large, not speed at any specific n."},
    {"id": "more_lines_slower", "concept_id": "time_complexity", "misconception": "Adding more lines of code makes a function slower in Big O terms.", "correction": "Big O measures how many operations grow WITH THE INPUT SIZE, not the line count. A 100-line function that does one pass over the data is O(n); a 5-line function with a nested loop is O(n^2)."},
    {"id": "try_except_syntax", "concept_id": "exceptions_basics", "misconception": "try/except is used to catch syntax errors in your code.", "correction": "Syntax errors are detected BEFORE the program runs and cannot be caught with try/except. try/except handles RUNTIME exceptions: ValueError, TypeError, KeyError, ZeroDivisionError, IOError, and so on."}
  ]
}
```

Key functions in `socratix/misconceptions.py`:

```
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
DEFAULT_COLLECTION = "socratix_misconceptions"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.7  # cosine distance; results above this are "no match"


def build_misconception_db(
    data_path: Path,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> chromadb.Collection:
    # 1. Load misconceptions.json.
    # 2. chromadb.PersistentClient(path=str(persist_dir)).
    # 3. Use SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL).
    # 4. client.get_or_create_collection(name, embedding_function=ef).
    # 5. For each misconception: add(documents=[misconception_text],
    #    metadatas=[{"concept_id": ..., "correction": ..., "id": ...}],
    #    ids=[misconception_id]).
    # 6. Return the collection.

def get_collection(
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> chromadb.Collection:
    # Open existing persistent collection without re-seeding.

def find_correction(
    collection: chromadb.Collection,
    misconception_text: str,
    *,
    top_k: int = 1,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict | None:
    # Returns {"misconception": ..., "correction": ..., "concept_id": ...,
    #          "distance": ...} or None if the best match is above threshold.
```

Embedding function setup:

```
from chromadb.utils import embedding_functions
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
```

Test plan (`scripts/test_phase6.py`):

1. Build the DB. First run will download the sentence-transformers
   model (~80 MB) - that is expected.
2. Query "I think recursion needs a for loop inside it" - expect a
   match against `recursion_loop_inside`, distance < 0.5.
3. Query "I believe list indices cannot be negative" - expect a
   match against `negative_index_error`.
4. Query "What is the syntax of a class definition?" - expect None
   (no good match in the dataset).
5. Re-open the collection in a fresh `get_collection` call - assert
   document count is 20 (persistence works).

Local tradeoff note: `all-MiniLM-L6-v2` is a strong general-purpose
embedding model but trained on web text, not student responses. For
ambiguous misconceptions the top-1 match may be semantically close but
referring to a different concept; the threshold of 0.7 helps but is
heuristic. If you see bad matches in practice, lower the threshold to
0.5 or add more misconceptions to widen coverage.

---

## PHASE 7 - Teaching agent - REMAINING

Goal: for each concept in the gap path, generate an explanation
calibrated to (a) the student's known concepts (analogies),
(b) any relevant misconception correction from Phase 6. After
explaining, ask the student to apply it. If they still do not
understand, retry ONCE with a different analogy, then flag for human
review.

Files to create:

- Append to `socratix/prompts.py` (do not create a new file):
  - `TEACHING_EXPLANATION`
  - `TEACHING_RETRY_EXPLANATION`
  - `UNDERSTANDING_CHECK`
  - `UNDERSTANDING_CHECK_STRICT`
- `socratix/teaching_agent.py`
- `scripts/test_phase7.py`

Prompts to add to `socratix/prompts.py`:

```
TEACHING_EXPLANATION = """You are a Python tutor explaining ONE concept to a student.

You will be given:
- The concept to teach (name and description).
- A short list of concepts the student has already mastered. Use these for analogies.
- Optionally, a misconception correction relevant to this concept.

Rules:
- Write 3 to 5 sentences of clear, friendly prose.
- Use ONE analogy drawn from a concept the student already knows. Name the analogy concept explicitly.
- Include ONE small concrete Python code example (3 to 6 lines), inside a fenced code block.
- If a misconception correction is provided: do not repeat the wrong belief verbatim. Gently redirect to the right understanding.
- After the explanation, ask the student ONE specific application question (write a tiny snippet, predict what a 2-line example would print, etc).
- No headers, no preamble, no bullet lists. Plain prose plus one code block.
"""

TEACHING_RETRY_EXPLANATION = """The student did not understand the previous explanation. Try AGAIN with:
- A DIFFERENT analogy than before. Name the previous analogy and explain you are switching.
- Simpler language and a smaller code example (2 to 4 lines).
- Keep all the original rules: explanation + code block + one specific application question at the end.
"""

UNDERSTANDING_CHECK = """Classify whether the student's follow-up response shows they NOW understand the concept that was just explained.

Categories:
- "understood": the response shows accurate application, accurate description, or a correct answer to the application question. Partial-but-correct counts.
- "still_struggling": the response is confused, repeats the original misconception, gives an incorrect answer, or asks for more clarification without engaging.

Output rules:
- Respond with a SINGLE JSON object, no markdown.
- Schema: {"verdict": "understood" | "still_struggling", "confidence": <float 0..1>, "rationale": "<one short sentence>"}.
"""

UNDERSTANDING_CHECK_STRICT = """Output VALID JSON only, no markdown.
Schema: {"verdict": "understood" | "still_struggling", "confidence": <float 0..1>, "rationale": "<one sentence>"}.
"""
```

Key functions in `socratix/teaching_agent.py`:

```
@dataclass(frozen=True)
class TeachResult:
    concept_id: str
    success: bool                      # True if understood within max_retries
    retries_used: int                  # 0, 1, or 2
    flagged_for_review: bool           # True iff success is False
    transcript: list[dict]             # full explanation/response/verdict trail


def generate_explanation(
    concept_attrs: dict,
    known_concept_names: list[str],
    misconception_correction: str | None,
    *,
    retry_n: int = 0,
    previous_analogy: str | None = None,
) -> str:
    # retry_n == 0 uses TEACHING_EXPLANATION; retry_n >= 1 uses
    # TEACHING_RETRY_EXPLANATION and includes previous_analogy in the
    # user message.

def check_understanding(
    concept_attrs: dict,
    explanation: str,
    student_followup: str,
    *,
    max_retries: int = 2,
) -> dict:
    # Reuse the retry-with-stricter-prompt pattern from
    # diagnostic.classify_response. Returns
    # {"verdict": ..., "confidence": ..., "rationale": ...}.
    # On exhausted retries, return verdict="still_struggling",
    # confidence=0.0.

def teach_concept(
    graph: nx.DiGraph,
    profile: StudentProfile,
    concept_id: str,
    misconception_collection: chromadb.Collection | None,
    *,
    student_response_provider: Callable[[str], str],
    max_retries: int = 2,
) -> TeachResult:
    # 1. Resolve concept_attrs from graph.nodes[concept_id].
    # 2. Look up known_concept_names = profile concepts with
    #    status == 'confirmed_known'. If empty, fall back to a couple of
    #    foundational concept names so analogies do not break.
    # 3. If profile has a stored misconception_summary for this concept,
    #    query the Chroma collection (find_correction) for a correction.
    # 4. For attempt in 0..max_retries:
    #      explanation = generate_explanation(..., retry_n=attempt, ...).
    #      student_followup = student_response_provider(explanation).
    #      verdict = check_understanding(..., student_followup).
    #      Append all three to transcript.
    #      If verdict.verdict == "understood":
    #          apply a synthetic DiagnosticResult marking the concept
    #          confirmed_known in profile + graph; return TeachResult
    #          (success=True, retries_used=attempt, flagged=False).
    #    All attempts failed -> mark_for_review(profile, concept_id);
    #    return TeachResult (success=False, retries_used=max_retries,
    #    flagged=True).
```

The `student_response_provider` callback exists so the same function
can drive a Streamlit UI (callback collects input from
`st.chat_input`), a CLI test (callback reads `input()`), or an automated
test (callback returns canned strings).

Local tradeoff: 8B models often produce shallow or inappropriate
analogies. The retry path's "use a DIFFERENT analogy" instruction is the
primary mitigation. After 2 retries, flagging for human review is the
right call - do not loop further.

Test plan (`scripts/test_phase7.py`):

1. Mark `variables`, `function_basics`, `return_values`,
   `if_statements` as `confirmed_known` in a fresh profile.
2. With a stub that returns "I get it: a recursive function calls
   itself on a smaller version of the input" as the followup, call
   `teach_concept("recursion_basics", ...)`. Assert
   `result.success == True`, retries_used == 0, profile shows
   `recursion_basics` as `confirmed_known`.
3. With a stub that always returns "I still do not get it", call
   `teach_concept("recursion_basics", ...)`. Assert
   `result.success == False`, retries_used == 2,
   `flagged_for_review == True`, profile's `needs_review` contains
   `recursion_basics`, transcript has 3 explanation/followup/verdict
   rounds.

---

## PHASE 8 - Progress persistence - REMAINING

Goal: SQLite-backed durability so progress survives across days, not
just within one session. Phase 4 JSON profiles are the in-session
working copy; SQLite is the long-term store.

Files to create:

- `socratix/persistence.py`
- `scripts/test_phase8.py`
- `socratix.sqlite` (gitignored, created on first run)

Schema (idempotent CREATE statements - safe to run on every startup):

```sql
CREATE TABLE IF NOT EXISTS students (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      TEXT NOT NULL,
    target_concept  TEXT,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    FOREIGN KEY (student_id) REFERENCES students(id)
);

CREATE TABLE IF NOT EXISTS concept_status (
    student_id              TEXT NOT NULL,
    concept_id              TEXT NOT NULL,
    status                  TEXT NOT NULL,   -- known | unknown | misconception | unassessed
    diagnostic_label        TEXT,            -- confirmed_known | confirmed_unknown | misconception_detected
    confidence              REAL NOT NULL,
    misconception_summary   TEXT,
    rationale               TEXT,
    last_seen               TEXT NOT NULL,
    PRIMARY KEY (student_id, concept_id),
    FOREIGN KEY (student_id) REFERENCES students(id)
);

CREATE TABLE IF NOT EXISTS conversation (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    concept_id  TEXT,
    role        TEXT NOT NULL,           -- system | student
    kind        TEXT NOT NULL,           -- question | response | classification | explanation | followup | understanding_check
    content     TEXT NOT NULL,
    payload     TEXT,                    -- JSON blob for structured kinds
    timestamp   TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_conv_session     ON conversation(session_id);
CREATE INDEX IF NOT EXISTS idx_status_student   ON concept_status(student_id);
```

Key functions in `socratix/persistence.py`:

```
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "socratix.sqlite"


def init_db(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    # Create tables if missing. Return a connection with row_factory =
    # sqlite3.Row.

def upsert_student(conn, student_id: str) -> None: ...

def start_session(conn, student_id: str, target_concept: str | None) -> int: ...

def end_session(conn, session_id: int) -> None: ...

def save_concept_status(
    conn,
    student_id: str,
    concept_id: str,
    status: str,                       # graph-level: known/unknown/misconception/unassessed
    diagnostic_label: str | None,      # diagnostic-level: confirmed_known/etc
    confidence: float,
    misconception_summary: str | None,
    rationale: str | None,
) -> None:
    # Use INSERT ... ON CONFLICT(student_id, concept_id) DO UPDATE.

def load_student_state(conn, student_id: str) -> dict[str, dict]:
    # Returns {concept_id: {status, confidence, rationale, ...}}.

def append_conversation(
    conn,
    session_id: int,
    concept_id: str | None,
    role: str,
    kind: str,
    content: str,
    payload: dict | None = None,
) -> None:
    # payload is JSON-encoded before insert.

def student_profile_to_db(conn, profile: StudentProfile, session_id: int) -> None:
    # Write/update student row, write all concept_statuses, append any
    # new conversation entries (track which entries are already
    # persisted using a high-water-mark on len(conversation_history)).

def db_to_student_profile(conn, student_id: str, target_concept: str) -> StudentProfile:
    # Reconstruct a StudentProfile from the student/concept_status tables.
    # Conversation history is loaded only on demand for UI; pass the
    # caller a flag to include or exclude it.
```

Test plan (`scripts/test_phase8.py`):

1. `init_db` on a fresh path. Assert all four tables exist
   (`PRAGMA table_info`).
2. `start_session("test_student_002", "recursion")` returns an int.
3. `save_concept_status` for three concepts.
4. `append_conversation` for five entries (mix of kinds).
5. `end_session`.
6. Close connection. Open a fresh connection.
7. `load_student_state` returns the three concept statuses unchanged.
8. Round-trip a StudentProfile via `student_profile_to_db` then
   `db_to_student_profile`. Assert value-equal (modulo timestamps,
   which may have second-level precision after JSON round-trip).
9. Edge case: re-saving the same concept_id for the same student must
   UPDATE not INSERT (no PK violation).

---

## PHASE 9 - Streamlit frontend - REMAINING

Goal: clean chat UI for the diagnostic and teaching dialogue, sidebar
showing the live Pyvis graph with color coding, target-topic picker.

File: `app.py` at the repo root.

requirements.txt addition: `streamlit==1.39.0`.

Layout:

- Sidebar (left):
  - Text input: student ID (defaults to "student").
  - Selectbox: target concept (populate from
    `[concepts.json -> concepts[].id]`, show `name` as label).
  - Button: "Start session".
  - Live Pyvis embed re-rendered every interaction.
  - Legend: green / red / orange / gray status colors.
  - Stats: counts of known / unknown / misconception / unassessed.
  - "Mark concept for review" expander listing
    `profile.needs_review`.

- Main area (right):
  - Chat-style display reading from
    `st.session_state.profile.conversation_history`.
  - `st.chat_input("Your answer ...")` at the bottom.
  - Status banner at top showing current phase and concept:
    "Diagnosing: For Loops" / "Teaching: Recursion Basics
    (attempt 2 of 3)".

State management (use `st.session_state`):

```
state keys:
  profile: StudentProfile
  graph:   nx.DiGraph
  conn:    sqlite3.Connection (Phase 8)
  session_id: int
  mode:    "idle" | "diagnose" | "teach"
  current_concept: str | None
  current_question: str | None       # cache to avoid regen on rerun
  current_explanation: str | None    # cache for teach mode
  gap_path: list[str]                # FIFO; pop after each success
  teaching_attempts: dict[str, int]  # concept_id -> attempts used
  misconception_collection: chromadb.Collection
```

Lifecycle (Streamlit re-runs the whole script on each interaction;
guard with `st.session_state`):

1. App start: if `state.profile` is None, render the sidebar with the
   ID / target picker only.
2. "Start session" click:
   - `upsert_student`, `start_session`, build fresh
     `StudentProfile` or load from DB.
   - `load_and_build` graph + `sync_graph_from_profile`.
   - `find_gap_path(target)` -> `state.gap_path`.
   - Set `state.mode = "diagnose"`, `state.current_concept =
     gap_path[0]`, generate first question, store in
     `state.current_question`.
3. On student input:
   - If `mode == "diagnose"`:
     - `classify_response`. `apply_diagnostic_result` (also writes to
       SQLite via `save_concept_status` + `append_conversation`).
     - If `confirmed_known`: pop from `gap_path`. If empty, mode ->
       "idle" and show success. Else move to next concept, generate
       question.
     - Else: mode -> "teach", generate explanation, store in
       `state.current_explanation`.
   - If `mode == "teach"`:
     - `check_understanding`. Append to history + DB.
     - If "understood": mark concept known in profile/graph/DB. Pop
       from `gap_path`. Move to next concept (re-enter "diagnose").
     - Else: increment `state.teaching_attempts[current_concept]`. If
       under `MAX_RETRIES`, generate retry explanation; else
       `mark_for_review`, advance `gap_path`.
4. After every state change, re-render the Pyvis sidebar by writing
   the HTML to `output/concept_graph.html` and embedding with
   `streamlit.components.v1.html`.

Pyvis embedding pattern:

```
import streamlit as st
from streamlit.components.v1 import html

html_path = visualize_graph(graph, "output/concept_graph.html")
html(html_path.read_text(encoding="utf-8"), height=750, scrolling=True)
```

Performance notes:

- Re-running `visualize_graph` on every Streamlit rerun is cheap for 50
  nodes (<100 ms). No caching needed.
- Loading the ChromaDB collection on every rerun is NOT cheap. Wrap it
  in `@st.cache_resource`.
- Same for the sentence-transformers model (it is what
  ChromaDB caches under the hood; this is automatic).
- Wrap `init_db` and `load_and_build` in `@st.cache_resource` too.

Manual test plan (Phase 9 has no automated test script - the test IS
launching the UI and walking through it):

1. `streamlit run app.py`.
2. Enter student ID "demo", target "recursion_basics", click Start.
3. Answer the first diagnostic question. Confirm chat updates and
   sidebar graph re-colors.
4. Answer subsequent questions: at least one correctly, one with a
   misconception, one declining.
5. Confirm "Teaching" mode kicks in after misconception/unknown
   classifications.
6. Confirm session persists: kill the app, restart, enter same
   student ID, verify status survives.

---

## PHASE 10 - Polish and README - REMAINING

Goal: ship-quality README, finalized `requirements.txt`, screenshot
placeholders.

File: `README.md` at repo root.

Required sections:

1. What is Socratix? (one elevator-pitch paragraph).
2. How it differs from a generic chatbot tutor:
   - Builds a knowledge graph of YOUR understanding, not a flat chat.
   - Targets exact gaps via topological sort, not blanket explanations.
   - Matches your misconceptions semantically against a curated
     correction database.
   - Uses concepts YOU already know as analogies for new ones.
3. Tech stack: the table from section 2 of this doc, with an explicit
   note: "Zero paid APIs. Runs entirely on your laptop."
4. Setup:
   - Install Python 3.10 or newer.
   - Install Ollama from `https://ollama.com/download`.
   - `ollama pull llama3.1:8b` (or `ollama pull mistral:7b` if 8B is
     slow; then `set OLLAMA_MODEL=mistral:7b`).
   - `pip install -r requirements.txt`.
   - `python scripts/test_phase2.py` to verify Ollama is reachable.
   - `streamlit run app.py` to launch.
5. Project structure: the final file tree from section 5 of this doc.
6. Per-phase tests: a bullet list of `python scripts/test_phaseN.py`
   commands the user can run to verify each phase.
7. Screenshots: three placeholders.
   - `docs/screenshot_diagnostic.png` - chat in diagnostic mode.
   - `docs/screenshot_graph.png` - sidebar Pyvis graph with mixed status.
   - `docs/screenshot_teaching.png` - chat in teaching mode after a
     misconception.
8. Limitations:
   - 8B local model is the source of most quality variance.
   - Domain is Python only; concept graph is in `data/concepts.json`
     and can be edited / replaced.
   - Misconception coverage is the 20-entry seed; extend
     `data/misconceptions.json` and rerun Phase 6 init to expand.
9. License (MIT or similar). State explicitly: no model weights are
   shipped; Ollama and the sentence-transformers model are downloaded
   on the user's machine.

Final `requirements.txt`:

```
networkx==3.3
pyvis==0.3.2
requests==2.32.3
chromadb==0.5.5
sentence-transformers==3.0.1
streamlit==1.39.0
```

(Python 3.10 minimum; tested on 3.14.)

---

## 6. How to resume with a free or less-capable model

The trap is asking a weaker model to "build Socratix" or "do
phase 4". It will lose track. Use this workflow:

1. At the START of every new chat, paste:
   - Sections 1, 2, 3 of this document (overview, tech stack, working
     rules).
   - The "PHASE N" section for the phase you are working on.
   - The current `requirements.txt`.
   - The existing relevant files (e.g. for Phase 4 you need
     `concept_graph.py`, `diagnostic.py`, `prompts.py`).
2. Tell the model: "Implement Phase N. Do not modify any other file.
   Do not skip ahead. Write the runnable test script too."
3. Run the test. Paste the output back. If it fails, ask the model to
   fix THAT failure, not redesign anything.
4. Only after the test passes, start a new chat for the next phase.

What to NOT let the model do:

- Re-derive the concept graph or the misconception seed dataset. The
  20 misconceptions in Phase 6 of this doc are curated; the model will
  produce shallower ones.
- Re-engineer the prompts in `socratix/prompts.py`. They were tuned for
  the 8B model. Change ONE prompt at a time, ONLY when a test fails.
- Combine phases. Each phase has a single test target; merged phases
  are harder to debug.

Order of remaining work (strict): 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10.
Phase 9 depends on every phase before it. Do not start Phase 9 until
Phases 4-8 all have passing tests.

---

## 7. Common failure modes and quick fixes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ask_llm_json` raises `JSONDecodeError` | 8B model emitted prose around the JSON | Already handled in Phase 3 retry path. For new callers, copy the `classify_response` retry pattern. |
| ChromaDB query returns unrelated correction | Threshold too lenient | Lower `SIMILARITY_THRESHOLD` (Phase 6) from 0.7 to 0.5, or add more misconceptions to the seed. |
| Streamlit graph does not update | `st.session_state.graph` not mutated, or `visualize_graph` called with stale graph | Always pass `state.graph` directly; do not deep-copy. After each turn call `sync_graph_from_profile` then regenerate the HTML. |
| Ollama times out (>120 s) | Model is loading from disk for the first time, or running on CPU with little RAM | First call is slow; subsequent calls are fast. If still slow, `set OLLAMA_MODEL=mistral:7b`. |
| Teaching agent loops forever | `student_response_provider` always returns the same string | The hard cap is `max_retries=2`. After that, `flagged_for_review` must be True; verify by inspecting the TeachResult. |
| SQLite "database is locked" | Multiple writers to the same `socratix.sqlite` | In Streamlit, hold ONE connection in `st.session_state`. Do not open new connections in callbacks. |

---

## 8. Acceptance criteria for the finished project

When everything is built, this command sequence must work on a fresh
machine:

```
git clone <repo>
cd Socratix
# Install Ollama, pull llama3.1:8b
pip install -r requirements.txt
python scripts/test_phase1.py
python scripts/test_phase2.py
python scripts/test_phase3.py
python scripts/test_phase4.py
python scripts/test_phase5.py
python scripts/test_phase6.py
python scripts/test_phase7.py
python scripts/test_phase8.py
streamlit run app.py
```

All eight test scripts exit 0. The Streamlit app launches, accepts a
student ID and a target concept, conducts a real Socratic diagnostic,
teaches at least one concept via analogy, color-codes the sidebar
graph live, and persists state across restarts.

That is the finish line.

---

End of build plan.
