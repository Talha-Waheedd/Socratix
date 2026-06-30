"""ChromaDB-backed misconception correction retrieval.

Stores 20 common Python/CS misconceptions with targeted corrections, embedded
locally via sentence-transformers (``all-MiniLM-L6-v2``). Phase 7's teaching
agent queries this collection when a student's diagnostic response reveals a
misconception.

Local tradeoff: ``all-MiniLM-L6-v2`` is trained on general web text, not
student phrasing. Top-1 matches can be semantically close but refer to a
different concept. The :data:`SIMILARITY_THRESHOLD` is a heuristic gate;
lower it (e.g. to 0.5) if you see false negatives, or add more seed entries
to ``data/misconceptions.json`` to widen coverage.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import chromadb
from chromadb import Collection
from chromadb.utils import embedding_functions

DEFAULT_PERSIST_DIR: Path = (
    Path(__file__).resolve().parent.parent / "chroma_db"
)
DEFAULT_COLLECTION: str = "socratix_misconceptions"
DEFAULT_MISCONCEPTIONS_PATH: Path = (
    Path(__file__).resolve().parent.parent / "data" / "misconceptions.json"
)
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD: float = 0.45
"""Maximum cosine distance for a match to be returned.

ChromaDB cosine distance: 0.0 = identical, higher = less similar.
Results with distance above this threshold return ``None`` from
:func:`find_correction`. 0.45 rejects semantically weak matches (e.g.
unrelated student text that still lands near some seed entry) while
keeping strong hits (typical good matches are below 0.2).
"""


def _embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Return the shared sentence-transformers embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def load_misconceptions_data(
    data_path: Path | str = DEFAULT_MISCONCEPTIONS_PATH,
) -> list[dict[str, Any]]:
    """Load the misconceptions seed JSON.

    Args:
        data_path: Path to ``data/misconceptions.json``.

    Returns:
        List of misconception dicts with keys ``id``, ``concept_id``,
        ``misconception``, ``correction``.

    Raises:
        FileNotFoundError: If ``data_path`` does not exist.
        ValueError: If the file is missing a ``misconceptions`` list.
    """
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Misconceptions file not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    entries = payload.get("misconceptions")
    if not isinstance(entries, list):
        raise ValueError("misconceptions.json must contain a 'misconceptions' list.")

    return entries


def build_misconception_db(
    data_path: Path | str = DEFAULT_MISCONCEPTIONS_PATH,
    *,
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    reset: bool = True,
) -> Collection:
    """Create or re-seed the local ChromaDB misconception collection.

    First run downloads the sentence-transformers model (~80 MB). Subsequent
    runs reuse the cached weights.

    Args:
        data_path: Seed JSON path.
        persist_dir: Directory for ChromaDB persistent storage.
        collection_name: Chroma collection name.
        reset: If True, delete any existing collection before seeding so
            re-runs are idempotent. Set False to open without re-seeding.

    Returns:
        The populated ChromaDB :class:`~chromadb.Collection`.
    """
    entries = load_misconceptions_data(data_path)
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    ef = _embedding_function()

    if reset:
        try:
            client.delete_collection(name=collection_name)
        except (ValueError, chromadb.errors.NotFoundError):
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        collection.add(
            ids=[entry["id"] for entry in entries],
            documents=[entry["misconception"] for entry in entries],
            metadatas=[
                {
                    "entry_id": entry["id"],
                    "concept_id": entry["concept_id"],
                    "correction": entry["correction"],
                }
                for entry in entries
            ],
        )

    return collection


def get_collection(
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> Collection:
    """Open an existing persistent misconception collection.

    Does not re-seed. Call :func:`build_misconception_db` first if the
    collection does not exist yet.

    Args:
        persist_dir: ChromaDB storage directory.
        collection_name: Collection name.

    Returns:
        The existing ChromaDB collection.

    Raises:
        ValueError: If the collection does not exist.
    """
    persist_dir = Path(persist_dir)
    client = chromadb.PersistentClient(path=str(persist_dir))
    ef = _embedding_function()
    return client.get_collection(
        name=collection_name,
        embedding_function=ef,
    )


def find_correction(
    collection: Collection,
    misconception_text: str,
    *,
    top_k: int = 1,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict[str, Any] | None:
    """Find the closest matching misconception correction.

    Args:
        collection: Populated ChromaDB collection.
        misconception_text: Free-text description of the student's wrong
            belief (typically ``DiagnosticResult.misconception_summary``).
        top_k: Number of nearest neighbors to consider (usually 1).
        threshold: Maximum cosine distance for a valid match.

    Returns:
        A dict with keys ``misconception``, ``correction``, ``concept_id``,
        ``entry_id``, and ``distance``, or ``None`` if no match is close
        enough.
    """
    if not misconception_text.strip():
        return None

    results = collection.query(
        query_texts=[misconception_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids") or [[]]
    if not ids or not ids[0]:
        return None

    distance = float(results["distances"][0][0])
    if distance > threshold:
        return None

    metadata = results["metadatas"][0][0]
    document = results["documents"][0][0]

    return {
        "misconception": document,
        "correction": metadata.get("correction", ""),
        "concept_id": metadata.get("concept_id", ""),
        "entry_id": metadata.get("entry_id", ids[0][0]),
        "distance": distance,
    }


def reset_persist_dir(persist_dir: Path | str) -> None:
    """Remove a ChromaDB persist directory (for tests)."""
    path = Path(persist_dir)
    if path.exists():
        shutil.rmtree(path)
