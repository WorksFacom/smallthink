#!/usr/bin/env python3
"""Send the same questions to each API listed in a YAML config.

Writes one JSONL file per API under out_dir. Rows already in that file
are skipped, so a run can be continued after a failure. APIs run at the
same time; questions for one API stay sequential.

An API with `rag_from: <bot answers JSONL>` gets, for each question, the
passages the bot retrieved for that id, in the bot's prompt; questions the
bot refused for lack of passages are refused the same way without a call.
Without it the question goes closed-book.

  python compare.py config.yml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from google import genai
from google.genai import types
from groq import Groq

# Must stay identical to HEADER, build_prompt and NO_SOURCES_ANSWER in
# rag_bot/app/rag.py: RAG baselines get exactly the trained model's prompt.
RAG_HEADER = (
    "Voce e um assistente especializado em etica medica e direito medico brasileiro, "
    "com base nas normas do Conselho Federal de Medicina (CFM). Responda a pergunta "
    "usando exclusivamente os trechos normativos fornecidos, de forma objetiva, e cite "
    "ao final a fonte (resolucao e artigo) que fundamenta a resposta. Se os trechos "
    "fornecidos nao contiverem a informacao necessaria, diga que nao e possivel "
    "responder com base neles."
)
NO_SOURCES_ANSWER = "Não é possível responder a essa pergunta com base nos trechos normativos fornecidos."
THINK_RE = re.compile(r"<think>(.*?)(?:</think>|\Z)", re.S | re.I)


def rag_prompt(question: str, sources: list[dict[str, Any]]) -> str:
    if sources:
        chunks = "\n\n".join(f"[{p['citation']}]\n{p['text']}" for p in sources)
    else:
        chunks = "(nenhum trecho recuperado)"
    return f"{RAG_HEADER}\n\nTrechos normativos:\n\n{chunks}\n\nPergunta: {question.strip()}"


def split_think(content: str, reasoning: str | None) -> tuple[str, str | None]:
    """Answer without reasoning, and the reasoning (separate field or inline <think>)."""
    content = content or ""
    match = THINK_RE.search(content)
    if match:
        reasoning = reasoning or match.group(1)
        content = content[: match.start()] + content[match.end() :]
    return content.strip(), (reasoning or "").strip() or None


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: config must be a mapping")
    if not data.get("questions"):
        raise SystemExit(f"{path}: missing 'questions'")
    if not isinstance(data.get("apis"), list) or not data["apis"]:
        raise SystemExit(f"{path}: 'apis' must be a non-empty list")
    return data


def load_questions(path: Path, limit: int | None) -> list[Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        body = json.loads(text)
        if isinstance(body, list):
            items = body
        elif isinstance(body, dict) and isinstance(body.get("questions"), list):
            items = body["questions"]
        else:
            raise SystemExit(f"{path}: expected a JSON array or {{\"questions\": [...]}}")
    if limit is not None:
        items = items[:limit]
    return items


def question_of(item: Any) -> str:
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
        raw = str(problem)
        idx = raw.rfind(marker)
        text = raw[idx + len(marker) :] if idx != -1 else raw
        text = text.strip()
        if text:
            return text
    raise ValueError("item needs 'question' or 'problem'")


def item_id(item: Any, index: int) -> str:
    if isinstance(item, dict):
        raw = item.get("id") or item.get("qa_id") or item.get("scenario_id")
        if raw:
            return str(raw)
    return str(index)


def reference_of(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    ref = item.get("solution")
    if ref is None:
        ref = item.get("reference")
    if ref is None and item.get("question"):
        ref = item.get("answer")
    if ref is None:
        return None
    text = str(ref).strip()
    return text or None


def post_json(url: str, payload: dict[str, Any], timeout: int, headers: dict[str, str] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"request failed: {e.reason}") from e
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError("response was not a JSON object")
    return parsed


def call_bot(api: dict[str, Any], item: Any) -> dict[str, Any]:
    url = api["url"]
    timeout = int(api.get("timeout") or 600)
    body: dict[str, Any] = {"questions": [item]}
    if api.get("top_k") is not None:
        body["top_k"] = int(api["top_k"])
    payload = post_json(url, body, timeout)
    results = payload.get("results") or []
    if not results:
        raise RuntimeError(f"bot returned no results: {payload}")
    row = results[0]
    if row.get("error"):
        raise RuntimeError(str(row["error"]))
    return {
        "answer": row.get("answer"),
        "thought": row.get("thought"),
        "abstention": row.get("abstention"),
        "sources": row.get("sources") or [],
        "top_k": payload.get("top_k"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "model": payload.get("model"),
    }


def with_cfm_source(question: str) -> str:
    """Ask a closed-book model to name the CFM norm behind the answer."""
    return (
        "Responda em português do Brasil, de forma objetiva. "
        "A resposta deve trazer a fonte do Conselho Federal de Medicina "
        "(resolução e artigo, ou considerando) que fundamenta a resposta. "
        "Se não souber a norma, diga que não é possível indicar a fonte.\n\n"
        f"Pergunta: {question}"
    )


def call_groq(api: dict[str, Any], prompt: str) -> dict[str, Any]:
    """`params` (reasoning_effort, reasoning_format, top_p, max_completion_tokens, ...)
    go straight to the API. Reasoning comes back in message.reasoning (gpt-oss by
    default; Qwen with reasoning_format: parsed) or inline as <think> (raw)."""
    env_name = api.get("api_key_env") or "GROQ_API_KEY"
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(f"missing env {env_name}")
    client = Groq(api_key=key, timeout=int(api.get("timeout") or 120))
    started = time.time()
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=api["model"],
        temperature=float(api.get("temperature") or 0),
        **(api.get("params") or {}),
    )
    try:
        choice = chat_completion.choices[0]
        message = choice.message
    except (AttributeError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected Groq response: {chat_completion}") from e
    if choice.finish_reason == "length":
        raise RuntimeError("cut off at max_completion_tokens (finish_reason=length; reasoning tokens count)")
    answer, thought = split_think(message.content or "", getattr(message, "reasoning", None))
    if not answer:
        raise RuntimeError(f"empty answer (finish_reason={choice.finish_reason})")
    return {
        "answer": answer,
        "thought": thought,
        "elapsed_seconds": round(time.time() - started, 2),
        "model": api["model"],
    }


def call_ollama(api: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Ollama chat API (the model's own chat template). `think` turns on the
    reasoning mode of thinking-capable models; `options` go to Ollama as is."""
    url = (api.get("url") or "http://localhost:11434").rstrip("/") + "/api/chat"
    options = {"temperature": float(api.get("temperature") or 0), **(api.get("options") or {})}
    body: dict[str, Any] = {
        "model": api["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": options,
    }
    if api.get("think") is not None:
        body["think"] = bool(api["think"])
    started = time.time()
    data = post_json(url, body, int(api.get("timeout") or 600))
    if data.get("done_reason") == "length":
        raise RuntimeError("cut off at options.num_predict (done_reason=length; thinking tokens count)")
    message = data.get("message") or {}
    answer, thought = split_think(message.get("content") or "", message.get("thinking"))
    if not answer:
        raise RuntimeError(f"empty answer (done_reason={data.get('done_reason')})")
    return {
        "answer": answer,
        "thought": thought,
        "elapsed_seconds": round(time.time() - started, 2),
        "model": api["model"],
    }


def call_gemini(api: dict[str, Any], prompt: str) -> dict[str, Any]:
    env_name = api.get("api_key_env") or "GEMINI_API_KEY"
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(f"missing env {env_name}")
    timeout_s = int(api.get("timeout") or 120)
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=timeout_s * 1000),
    )
    started = time.time()
    interaction = client.interactions.create(
        model=api["model"],
        input=prompt,
        generation_config={"temperature": float(api.get("temperature") or 0)},
    )
    answer = interaction.output_text
    if not answer:
        raise RuntimeError(f"unexpected Gemini response: {interaction}")
    return {
        "answer": answer,
        "elapsed_seconds": round(time.time() - started, 2),
        "model": api["model"],
    }


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("id") is not None and not row.get("error"):
            done.add(str(row["id"]))
    return done


def load_rag_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Bot answer rows by id: the passages (and refusals) to reuse."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("id") is not None and not row.get("error"):
                rows[str(row["id"])] = row
    return rows


def call_with_context(api: dict[str, Any], question: str, rag_row: dict[str, Any] | None) -> dict[str, Any]:
    kind = api["type"]
    if rag_row is None:
        prompt, extra = with_cfm_source(question), {"context": "closed_book"}
    else:
        sources = rag_row.get("sources") or []
        extra = {"context": "rag", "sources": sources, "top_k": rag_row.get("top_k"), "abstention": None}
        if not sources and api.get("abstain_without_sources", True):
            return {**extra, "answer": NO_SOURCES_ANSWER, "thought": None, "abstention": "no_sources",
                    "elapsed_seconds": 0.0, "model": api.get("model")}
        prompt = rag_prompt(question, sources)
    if kind == "groq":
        result = call_groq(api, prompt)
    elif kind == "gemini":
        result = call_gemini(api, prompt)
    elif kind == "ollama":
        result = call_ollama(api, prompt)
    else:
        raise RuntimeError(f"unknown type '{kind}' (use bot, groq, gemini, or ollama)")
    return {**extra, **result}


def run_api(api: dict[str, Any], items: list[Any], out_path: Path, rag_rows: dict[str, dict[str, Any]] | None) -> None:
    name = api["name"]
    kind = api["type"]
    done = load_done(out_path)
    print(f"== {name} ({kind}{', rag' if rag_rows is not None else ''}) -> {out_path}", file=sys.stderr)
    with out_path.open("a", encoding="utf-8") as out:
        for index, item in enumerate(items):
            qid = item_id(item, index)
            if qid in done:
                continue
            question = None
            record: dict[str, Any] = {"id": qid, "api": name}
            ref = reference_of(item)
            if ref is not None:
                record["reference"] = ref
            try:
                question = question_of(item)
                record["question"] = question
                print(f"[{index + 1}/{len(items)}] {name} {qid}", file=sys.stderr, flush=True)
                if kind == "bot":
                    record.update(call_bot(api, item))
                else:
                    rag_row = None
                    if rag_rows is not None:
                        rag_row = rag_rows.get(qid)
                        if rag_row is None:
                            raise RuntimeError(f"no row for {qid} in {api['rag_from']}")
                    record.update(call_with_context(api, question, rag_row))
            except Exception as e:
                record["question"] = question
                record["error"] = str(e)
                print(f"[{index + 1}/{len(items)}] {name} {qid} error: {e}", file=sys.stderr)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask each configured API the same questions.")
    parser.add_argument("config", nargs="?", default="config.yml", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    questions_path = Path(cfg["questions"])
    if not questions_path.is_file():
        questions_path = (args.config.parent / questions_path).resolve()
    if not questions_path.is_file():
        raise SystemExit(f"questions file not found: {cfg['questions']}")
    limit = cfg.get("limit")
    items = load_questions(questions_path, int(limit) if limit else None)
    if not items:
        raise SystemExit("no questions")

    out_dir = Path(cfg.get("out_dir") or "answers")
    if not out_dir.is_absolute():
        out_dir = (args.config.parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[dict[str, Any], Path, dict[str, dict[str, Any]] | None]] = []
    seen: set[str] = set()
    for api in cfg["apis"]:
        name = api.get("name")
        if not name or not api.get("type"):
            raise SystemExit("each api needs 'name' and 'type'")
        if name in seen:
            raise SystemExit(f"duplicate api name '{name}'")
        seen.add(name)
        rag_rows = None
        if api.get("rag_from"):
            if api["type"] == "bot":
                raise SystemExit(f"{name}: rag_from is for non-bot APIs")
            rag_path = Path(api["rag_from"])
            if not rag_path.is_file():
                rag_path = (args.config.parent / rag_path).resolve()
            if not rag_path.is_file():
                raise SystemExit(f"{name}: rag_from file not found: {api['rag_from']}")
            rag_rows = load_rag_rows(rag_path)
        jobs.append((api, out_dir / f"{name}.jsonl", rag_rows))

    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(run_api, api, items, path, rag_rows) for api, path, rag_rows in jobs]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                errors.append(e)
    if errors:
        detail = "; ".join(str(e) for e in errors)
        raise SystemExit(f"{len(errors)} api run(s) failed: {detail}")
    print(f"wrote {len(jobs)} files in {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
