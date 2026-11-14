"""Small stdlib helpers shared by the data-generation scripts.

HTTP calls to a local Ollama server (OpenAI-compatible ``/v1`` endpoints)
and to Qdrant's REST API, plus text checks used as data filters.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

THINK_RE = re.compile(r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", re.IGNORECASE)

# Function words only; "a", "as", "e", "no", "me" are left out because they
# are common in both languages.
PT_WORDS = {
    "de", "o", "que", "do", "da", "em", "um", "uma", "para", "é", "com", "não",
    "os", "se", "na", "por", "mais", "dos", "das", "como", "mas", "ao", "aos",
    "seu", "sua", "seus", "suas", "ou", "quando", "nos", "nas", "já", "também",
    "pelo", "pela", "pelos", "pelas", "até", "isso", "ela", "ele", "entre",
    "sem", "mesmo", "esse", "essa", "este", "esta", "qual", "deve", "pode",
    "ser", "são", "está", "médico", "paciente", "à", "às", "nº",
}
EN_WORDS = {
    "the", "of", "and", "to", "in", "is", "that", "for", "it", "was", "with",
    "be", "by", "on", "not", "this", "are", "or", "from", "at", "which", "but",
    "have", "an", "they", "you", "were", "there", "been", "if", "their",
    "would", "what", "will", "should", "must", "can", "patient", "physician",
    "doctor", "medical", "need", "let", "okay",
}


def post_json(url: str, payload: dict[str, Any], timeout: int, headers: dict[str, str] | None = None) -> Any:
    req_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=req_headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    timeout: int,
    json_mode: bool = False,
    extra: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if extra:
        payload.update(extra)
    body = post_json(base_url.rstrip("/") + "/chat/completions", payload, timeout)
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"malformed chat response: {body!r}") from e


def embed(base_url: str, model: str, texts: list[str], timeout: int) -> list[list[float]]:
    body = post_json(base_url.rstrip("/") + "/embeddings", {"model": model, "input": texts}, timeout)
    items = body.get("data") or []
    if len(items) != len(texts):
        raise RuntimeError(f"embedding count mismatch: got {len(items)}, expected {len(texts)}")
    return [it["embedding"] for it in items]


def qdrant_query(
    qdrant_url: str,
    collection: str,
    vector: list[float],
    limit: int,
    timeout: int,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Nearest points as [{"passage_id", "score"}]."""
    body = post_json(
        f"{qdrant_url.rstrip('/')}/collections/{collection}/points/query",
        {"query": vector, "limit": limit, "with_payload": ["passage_id"]},
        timeout,
        headers={"api-key": api_key} if api_key else None,
    )
    points = (body.get("result") or {}).get("points") or []
    return [
        {"passage_id": (p.get("payload") or {}).get("passage_id"), "score": p.get("score")}
        for p in points
    ]


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def extract_json(raw: str) -> str:
    """Outermost {...} span of a reply, ignoring reasoning blocks and code fences."""
    s = re.sub(r"```(?:json)?", "", strip_think(raw)).strip()
    start, end = s.find("{"), s.rfind("}")
    return s[start : end + 1] if start != -1 and end > start else s


def words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", (text or "").lower())


def looks_portuguese(text: str) -> bool:
    toks = words(text)
    pt = sum(t in PT_WORDS for t in toks)
    en = sum(t in EN_WORDS for t in toks)
    if len(toks) < 8:
        return pt > en
    return pt >= 3 and pt >= 2 * en
