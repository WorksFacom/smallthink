#!/usr/bin/env python3
"""HTTP API: batch of questions → RAG passages → fine-tuned model.

POST /v1/answers with a JSON list of questions. Each item is answered with
the same retrieval + CoT path as the terminal chat, so the output can be
compared with other LLMs on the same questions.

Accepted bodies
---------------
{"questions": ["...?", {"id": "cfm_0", "question": "...?"}]}

or a raw array, including rows of data_pipeline/data/rag_problems.jsonl.
A ``problem`` field (full training prompt) is reduced to the text after
``Pergunta:`` so retrieval embeds the question, not the stuffed passages.
``solution`` / ``answer`` on the input is echoed as ``reference``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from app import rag

MAX_BODY = 8 * 1024 * 1024
HOST = os.environ.get("API_HOST", "0.0.0.0")
PORT = int(os.environ.get("API_PORT", "8000"))


def _question_of(item: Any) -> str:
    if isinstance(item, str):
        text = item.strip()
        if not text:
            raise ValueError("empty question")
        return text
    if not isinstance(item, dict):
        raise ValueError("each item must be a string or an object")
    if item.get("question"):
        return str(item["question"]).strip()
    problem = item.get("problem")
    if problem:
        marker = "Pergunta:"
        idx = str(problem).rfind(marker)
        text = str(problem)[idx + len(marker) :] if idx != -1 else str(problem)
        text = text.strip()
        if text:
            return text
    raise ValueError("item needs 'question' or 'problem'")


def _normalize(body: Any) -> tuple[list[Any], int | None]:
    top_k = None
    if isinstance(body, list):
        items = body
    elif isinstance(body, dict):
        if "questions" not in body:
            raise ValueError("object body needs a 'questions' array")
        items = body["questions"]
        if not isinstance(items, list):
            raise ValueError("'questions' must be an array")
        if body.get("top_k") is not None:
            top_k = int(body["top_k"])
            if not 1 <= top_k <= 5:
                raise ValueError("top_k must be between 1 and 5")
    else:
        raise ValueError("body must be an array or an object with 'questions'")
    if not items:
        raise ValueError("questions is empty")
    return items, top_k


def _reference(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    ref = item.get("solution")
    if ref is None:
        ref = item.get("reference")
    if ref is None and "problem" in item:
        # training rows store the gold answer in "solution"; generated QA uses "answer"
        # alongside "question", so only echo "answer" when it is not the question itself
        ref = item.get("answer") if item.get("question") else None
    if ref is None:
        return None
    text = str(ref).strip()
    return text or None


def _item_id(item: Any, index: int) -> str:
    if isinstance(item, dict):
        raw = item.get("id") or item.get("qa_id") or item.get("scenario_id")
        if raw:
            return str(raw)
    return str(index)


def run_batch(body: Any) -> dict[str, Any]:
    items, top_k = _normalize(body)
    qdrant = rag.client()
    results: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        record: dict[str, Any] = {"id": _item_id(item, index)}
        ref = _reference(item)
        if ref is not None:
            record["reference"] = ref
        question = None
        try:
            question = _question_of(item)
            print(f"[{index + 1}/{len(items)}] {record['id']}", file=sys.stderr, flush=True)
            answered = rag.answer_question(qdrant, question, k=top_k)
            record.update(answered)
        except Exception as e:
            low = str(e).lower()
            if "collection" in low and ("not found" in low or "doesn't exist" in low):
                raise
            record["question"] = question
            record["error"] = str(e)
            print(f"[{index + 1}/{len(items)}] {record['id']} error: {e}", file=sys.stderr)
        results.append(record)
    return {
        "system": "trained+rag",
        "model": rag.GEN_MODEL,
        "embed_model": rag.EMBED_MODEL,
        "collection": rag.COLLECTION,
        "top_k": top_k if top_k is not None else rag.TOP_K,
        "count": len(results),
        "results": results,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[api] {self.address_string()} {fmt % args}", file=sys.stderr)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send(
                200,
                {
                    "ok": True,
                    "model": rag.GEN_MODEL,
                    "embed_model": rag.EMBED_MODEL,
                    "collection": rag.COLLECTION,
                    "qdrant": rag.QDRANT_URL,
                },
            )
            return
        if path == "/":
            self._send(
                200,
                {
                    "service": "cfm-trained-rag",
                    "post": "/v1/answers",
                    "body": {"questions": ["pergunta?"], "top_k": 2},
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/answers", "/v1/answer"):
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0 or length > MAX_BODY:
            self._send(400, {"error": f"Content-Length must be 1..{MAX_BODY}"})
            return
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
            payload = run_batch(body)
        except json.JSONDecodeError as e:
            self._send(400, {"error": f"invalid json: {e}"})
            return
        except ValueError as e:
            self._send(400, {"error": str(e)})
            return
        except Exception as e:
            low = str(e).lower()
            if "collection" in low and ("not found" in low or "doesn't exist" in low):
                self._send(
                    503,
                    {
                        "error": (
                            f"collection '{rag.COLLECTION}' is missing. "
                            "Run: docker compose --profile index run --rm index"
                        )
                    },
                )
                return
            self._send(502, {"error": str(e)})
            return
        self._send(200, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="CFM trained+RAG answer API")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), Handler)
    print(
        f"listening on http://{args.host}:{args.port}  "
        f"model={rag.GEN_MODEL} collection={rag.COLLECTION}",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(file=sys.stderr)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
