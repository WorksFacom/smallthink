"""Retrieve CFM passages and answer with the fine-tuned Ollama model.

The user prompt matches the training problems built by
``data_pipeline/build_rag_problems.py``: a fixed header, then
``[citation]\\ntext`` blocks, then ``Pergunta:``.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Iterator

from qdrant_client import QdrantClient

HEADER = (
    "Voce e um assistente especializado em etica medica e direito medico brasileiro, "
    "com base nas normas do Conselho Federal de Medicina (CFM). Responda a pergunta "
    "usando exclusivamente os trechos normativos fornecidos, de forma objetiva, e cite "
    "ao final a fonte (resolucao e artigo) que fundamenta a resposta. Se os trechos "
    "fornecidos nao contiverem a informacao necessaria, diga que nao e possivel "
    "responder com base neles."
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://host.docker.internal:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen3-4b-cfm-cot")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
COLLECTION = os.environ.get("COLLECTION", "cfm_passages")
TOP_K = int(os.environ.get("TOP_K", "3"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.56"))
NUM_CTX = int(os.environ.get("NUM_CTX", "4096"))
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "1536"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
# When no passage clears MIN_SCORE, refuse without calling the model: training
# never showed an empty passage block, so the model answers from memory there.
ABSTAIN_WITHOUT_SOURCES = os.environ.get("ABSTAIN_WITHOUT_SOURCES", "1").lower() not in ("0", "false", "no")
NO_SOURCES_ANSWER = (
    "Não é possível responder a essa pergunta com base nos trechos normativos fornecidos."
)
# The trained format ends at </Output>; nothing after it is part of the answer.
STOP = ["</Output>"]
OUTPUT_RE = re.compile(r"<output>(.*?)(?:</output>|\Z)", re.S | re.I)
# Opening and closing tags do not always match: Qwen3's native <think> can be
# closed by the trained </Thought>.
THOUGHT_RE = re.compile(r"<(?:think|thought)>(.*?)(?:</(?:think|thought)>|(?=<output>)|\Z)", re.S | re.I)


def client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def embed(text: str, timeout: int = 120) -> list[float]:
    body = json.dumps({"model": EMBED_MODEL, "input": [text]}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"embedding failed ({e.code}): {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach Ollama at {OLLAMA_URL}: {e.reason}") from e
    return data["data"][0]["embedding"]


def retrieve(qdrant: QdrantClient, question: str, k: int | None = None) -> list[dict[str, Any]]:
    k = TOP_K if k is None else k
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=embed(question),
        limit=k,
        with_payload=True,
    ).points
    out: list[dict[str, Any]] = []
    for hit in hits:
        if hit.score is not None and hit.score < MIN_SCORE:
            continue
        payload = dict(hit.payload or {})
        payload["score"] = hit.score
        out.append(payload)
    return out


def build_prompt(question: str, passages: list[dict[str, Any]]) -> str:
    if passages:
        chunks = "\n\n".join(f"[{p['citation']}]\n{p['text']}" for p in passages)
    else:
        chunks = "(nenhum trecho recuperado)"
    return f"{HEADER}\n\nTrechos normativos:\n\n{chunks}\n\nPergunta: {question.strip()}"


def _generate_request(prompt: str, stream: bool) -> urllib.request.Request:
    body = json.dumps(
        {
            "model": GEN_MODEL,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": TEMPERATURE,
                "num_ctx": NUM_CTX,
                "num_predict": NUM_PREDICT,
                "stop": STOP,
            },
        }
    ).encode("utf-8")
    return urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def generate(prompt: str, timeout: int = 600) -> str:
    req = _generate_request(prompt, stream=False)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"generation failed ({e.code}): {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach Ollama at {OLLAMA_URL}: {e.reason}") from e
    response = data.get("response") or ""
    # Ollama moves reasoning to `thinking` for templates it detects as thinking-capable.
    thinking = (data.get("thinking") or "").strip()
    if thinking:
        return f"<think>{thinking}</think>\n{response}"
    return response


def generate_stream(prompt: str, timeout: int = 600) -> Iterator[str]:
    req = _generate_request(prompt, stream=True)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"generation failed ({e.code}): {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach Ollama at {OLLAMA_URL}: {e.reason}") from e
    try:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            obj = json.loads(line)
            piece = obj.get("response") or ""
            if piece:
                yield piece
            if obj.get("done"):
                break
    finally:
        resp.close()


def unload(model: str = GEN_MODEL, timeout: int = 10) -> None:
    """Ask Ollama to evict ``model`` from memory now (``keep_alive: 0``).

    Best effort: failures are ignored so shutdown is never blocked.
    """
    body = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except (urllib.error.URLError, TimeoutError, OSError):
        return


def parse_answer(raw: str) -> tuple[str, str | None]:
    """Split reasoning from the reply.

    Trained outputs use ``<Thought>`` / ``<Output>``; Qwen3 also emits
    ``<think>``, and either opening tag may be closed by either closing tag.
    Generation stops at ``</Output>`` (Ollama drops the stop string), so the
    closing tag is optional. Ollama's ``thinking`` field can also swallow the
    whole ``<Output>`` block; then the answer is taken from the raw text.
    """
    text = raw.strip()
    thought = None
    match = THOUGHT_RE.search(text)
    if match:
        thought = match.group(1)
        text = (text[: match.start()] + text[match.end() :]).strip()
    output = OUTPUT_RE.search(text) or OUTPUT_RE.search(raw)
    if thought:
        thought = re.sub(r"^\s*(?:<(?:think|thought)>\s*)+", "", thought, flags=re.I)
        thought = re.split(r"</?(?:think|thought)>|<output>", thought, maxsplit=1, flags=re.I)[0].strip() or None
    answer = output.group(1).strip() if output else text
    return answer, thought


def answer_question(qdrant: QdrantClient, question: str, k: int | None = None) -> dict[str, Any]:
    """Retrieve passages and generate one answer. Raises RuntimeError on failure."""
    started = time.time()
    passages = retrieve(qdrant, question, k=k)
    if not passages and ABSTAIN_WITHOUT_SOURCES:
        answer, thought, abstention = NO_SOURCES_ANSWER, None, "no_sources"
    else:
        answer, thought = parse_answer(generate(build_prompt(question, passages)))
        abstention = None
    return {
        "question": question,
        "answer": answer,
        "thought": thought,
        "abstention": abstention,
        "sources": sources_of(passages),
        "elapsed_seconds": round(time.time() - started, 2),
    }


def sources_of(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "citation": p.get("citation"),
            "passage_id": p.get("passage_id"),
            "kind": p.get("kind"),
            "score": p.get("score"),
            "text": p.get("text"),
        }
        for p in passages
    ]
