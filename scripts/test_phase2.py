"""Phase 2 smoke test: confirm the local Ollama instance answers ask_llm().

Run from the repository root:

    python scripts/test_phase2.py

Prerequisites:
    1. Install Ollama:           https://ollama.com/download
    2. Start the daemon:         `ollama serve` (or open the desktop app)
    3. Pull the default model:   `ollama pull llama3.1:8b`
       (or `ollama pull mistral:7b` and set OLLAMA_MODEL=mistral:7b)

The script reports each check it performs and exits non-zero on the first
failure with an actionable hint, so you can fix the environment before
moving on to Phase 3 (which depends on this wrapper).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from socratix.llm import (  # noqa: E402  (sys.path tweak above)
    DEFAULT_MODEL,
    OLLAMA_BASE_URL,
    OllamaError,
    OllamaModelMissingError,
    OllamaUnavailableError,
    ask_llm,
    is_ollama_running,
    list_local_models,
)

SYSTEM_PROMPT = (
    "You are a concise Python tutor. Reply in ONE short sentence (max 20 words). "
    "Plain prose only; do not use markdown, code fences, or bullet points."
)

USER_MESSAGE = "In one sentence, what is a Python variable?"


def _fail(msg: str, hint: str | None = None) -> int:
    print(f"[FAIL] {msg}", file=sys.stderr)
    if hint:
        print(f"       hint: {hint}", file=sys.stderr)
    return 1


def main() -> int:
    print("Socratix Phase 2 - local LLM smoke test")
    print("-" * 50)
    print(f"Ollama base URL : {OLLAMA_BASE_URL}")
    print(f"Default model   : {DEFAULT_MODEL}")
    print()

    print("[1/3] Checking Ollama is reachable ...")
    if not is_ollama_running():
        return _fail(
            f"Ollama is not responding on {OLLAMA_BASE_URL}.",
            "Start it with `ollama serve` (or open the Ollama desktop app), then retry.",
        )
    print("      OK")

    print(f"[2/3] Verifying model {DEFAULT_MODEL!r} is pulled ...")
    try:
        models = list_local_models()
    except OllamaError as exc:
        return _fail(str(exc))

    if DEFAULT_MODEL not in models:
        return _fail(
            f"Model {DEFAULT_MODEL!r} is not in your local list: {models or '[]'}",
            f"Pull it with `ollama pull {DEFAULT_MODEL}` "
            f"(or set OLLAMA_MODEL to a model you have).",
        )
    print(f"      OK  (local models: {', '.join(models)})")

    print("[3/3] Running a real ask_llm() call ...")
    start = time.perf_counter()
    try:
        reply = ask_llm(SYSTEM_PROMPT, USER_MESSAGE)
    except OllamaModelMissingError as exc:
        return _fail(str(exc))
    except OllamaUnavailableError as exc:
        return _fail(str(exc))
    except OllamaError as exc:
        return _fail(f"ask_llm raised: {exc}")
    elapsed = time.perf_counter() - start

    reply_stripped = reply.strip()
    if not reply_stripped:
        return _fail("Model returned an empty response.")

    print(f"      OK  ({elapsed:.1f}s)")
    print()
    print("System prompt : " + SYSTEM_PROMPT)
    print("User message  : " + USER_MESSAGE)
    print("Model reply   : " + reply_stripped)
    print()
    print("Phase 2 looks good. Ready to build the diagnostic agent in Phase 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
