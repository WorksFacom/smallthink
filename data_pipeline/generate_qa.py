#!/usr/bin/env python3
"""
Generate Q&A pairs (pt-BR) from CFM passages using a local Ollama
server (OpenAI-compatible chat-completions endpoint at /v1).

Defaults assume Ollama is running at http://localhost:11434/v1. Start a model
beforehand, e.g.: `ollama pull qwen2.5:3b && ollama run qwen2.5:3b`.

The prompt follows the corpus recipe of Athaydes et al. (2025, "Towards a
Corpus Methodology for LLMs in the Legal Domain"): varied question types,
self-contained questions, technical tone, mandatory legal reference, and
sampling penalties for lexical/topic diversity (temperature 0.8, frequency
penalty 0.5, presence penalty 0.8). Passages marked ``duplicate_of`` are
skipped, and so are articles that only approve, put in force or revoke
(``BOILERPLATE_RE``), which yield trivia questions, and articles suspended by
a court decision (``SUSPENDED_RE``), which are not rules in force.

Env:
  OLLAMA_BASE_URL   default http://localhost:11434/v1
  QA_MODEL          default qwen2.5:3b

Input:
  data/passages.jsonl  (produced by ingest_cfm.py)

Output:
  data/qa_pairs.jsonl  (one pair per line, with provenance)

The generator is resumable: lines already written for a given
(passage_id, pair_index) are skipped on re-runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROMPT_VERSION = "cfm-qa-v3"
BOILERPLATE_RE = re.compile(r"entram?\s+em\s+vigor|revoga|Aprovar\s+o\s+C[óo]digo", re.IGNORECASE)
# Approve / in-force / revoke clauses open the article; later mentions are real rules.
BOILERPLATE_HEAD_CHARS = 150
# Articles suspended by a court decision are not rules in force.
SUSPENDED_RE = re.compile(r"artigo\s+suspenso\s+por\s+decis[ãa]o\s+judicial", re.IGNORECASE)

SYSTEM_PROMPT = (
    "Você é um especialista em ética médica e direito médico brasileiro. "
    "Sua tarefa é gerar pares de pergunta e resposta em português do Brasil "
    "estritamente a partir do TRECHO fornecido de uma norma do Conselho Federal "
    "de Medicina (CFM).\n\n"
    "Regras obrigatórias:\n"
    "1. Use SOMENTE informações presentes no TRECHO. Não infira, não complete, "
    "não traga conhecimento externo.\n"
    "2. Escreva em português do Brasil, com acentuação correta e tom técnico.\n"
    "3. Varie os tipos de pergunta entre os pares: pergunta direta sobre a regra; "
    "situação prática (um médico ou paciente em um caso concreto); permissão ou "
    "proibição (\"O médico pode...?\", \"É permitido...?\"); condições e exceções "
    "(\"Em que casos...?\", \"Salvo quando...?\"), somente se o TRECHO tiver "
    "condições ou exceções; definição ou finalidade.\n"
    "4. Cada pergunta deve ser autocontida: nomeie o tema explicitamente e NÃO use "
    "expressões como \"neste trecho\", \"segundo o texto\", \"conforme o artigo acima\". "
    "A pergunta NÃO pode mencionar número de resolução, artigo ou considerando: "
    "escreva como um médico ou paciente perguntaria, sem citar a norma.\n"
    "5. Não repita a mesma pergunta com outras palavras. Não reformule a pergunta "
    "como resposta: a resposta deve informar a regra, incluindo condições e "
    "exceções presentes no TRECHO.\n"
    "6. Cada resposta deve ser autocontida e objetiva e terminar com uma linha "
    "no formato: Fonte: <citation>\n"
    "7. Não forneça aconselhamento clínico individual. O conteúdo é normativo.\n"
    "8. Produza exatamente N pares, onde N é informado pelo usuário. Se o TRECHO "
    "não tiver informação suficiente para N perguntas diferentes, retorne menos "
    "pares (ou lista vazia).\n"
    "9. Saída ESTRITA em JSON válido no formato: "
    "{\"pairs\": [{\"question\": \"...\", \"answer\": \"...\"}, ...]} "
    "sem texto fora do JSON."
)

USER_TEMPLATE = (
    "Gere {n} pares de pergunta-resposta em pt-BR, estritamente a partir do "
    "trecho a seguir.\n\n"
    "Citação canônica (use exatamente na linha Fonte): {citation}\n"
    "Tipo do trecho: {kind} ({label})\n\n"
    "TRECHO:\n"
    "\"\"\"\n{text}\n\"\"\"\n\n"
    "Responda SOMENTE com JSON no formato: "
    "{{\"pairs\": [{{\"question\": \"...\", \"answer\": \"...\"}}]}}"
)


def _strip_json(raw: str) -> str:
    """Best-effort: drop reasoning blocks and ```json fences, keep the {...} span."""
    s = re.sub(r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", "", raw, flags=re.IGNORECASE)
    s = re.sub(r"```(?:json)?", "", s).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        return s[start : end + 1]
    return s


def _call_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        **(sampling or {}),
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _extract_content(api_resp: dict[str, Any]) -> str:
    try:
        return api_resp["choices"][0]["message"]["content"] or ""
    except Exception as e:
        raise RuntimeError(f"Malformed API response: {api_resp!r}") from e


def _build_messages(
    passage: dict[str, Any],
    n_pairs: int,
) -> list[dict[str, str]]:
    user = USER_TEMPLATE.format(
        n=n_pairs,
        citation=passage.get("citation", "?"),
        kind=passage.get("kind", "?"),
        label=passage.get("label") or "-",
        text=passage.get("text", ""),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _dry_run_pairs(
    passage: dict[str, Any],
    n_pairs: int,
) -> list[dict[str, Any]]:
    out = []
    label = passage.get("label") or passage.get("kind")
    for i in range(n_pairs):
        ans = f"[DRY-RUN] Resposta {i + 1} baseada no trecho.\nFonte: {passage.get('citation')}"
        out.append(
            {
                "question": f"[DRY-RUN] Pergunta {i + 1} sobre {label}",
                "answer": ans,
            }
        )
    return out


def _generate_pairs_for_passage(
    passage: dict[str, Any],
    n_pairs: int,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    dry_run: bool,
    sampling: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if dry_run:
        return _dry_run_pairs(passage, n_pairs)

    messages = _build_messages(passage, n_pairs)

    try:
        resp = _call_chat(
            base_url=base_url,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            sampling=sampling,
        )
        content = _extract_content(resp)
        obj = json.loads(_strip_json(content))
        pairs = obj.get("pairs", [])
        if not isinstance(pairs, list):
            raise ValueError("`pairs` is not a list")
        cleaned: list[dict[str, Any]] = []
        for p in pairs:
            if not isinstance(p, dict):
                continue
            q = (p.get("question") or "").strip()
            a = (p.get("answer") or "").strip()
            if q and a:
                cleaned.append({"question": q, "answer": a})
        return cleaned
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"generation failed for {passage.get('passage_id')}: {e}"
        ) from e


def _is_boilerplate(passage: dict[str, Any]) -> bool:
    text = passage.get("text") or ""
    ctx = passage.get("context")
    if ctx and text.startswith(ctx):
        text = text[len(ctx):]
    head = " ".join(text.split())[:BOILERPLATE_HEAD_CHARS]
    return passage.get("kind") == "artigo" and bool(BOILERPLATE_RE.search(head))


def _load_passages(path: Path, include_kinds: set[str], keep_boilerplate: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            if p.get("kind") not in include_kinds or p.get("duplicate_of"):
                continue
            if not keep_boilerplate and _is_boilerplate(p):
                continue
            if SUSPENDED_RE.search(p.get("text") or ""):
                continue
            out.append(p)
    return out


def _model_slug(model: str) -> str:
    """Filesystem/ID-safe slug for a model tag like 'qwen2.5:3b' -> 'qwen2.5-3b'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"


def _existing_keys(output: Path) -> set[tuple[str, str]]:
    """Return {(passage_id, model)} already present in the output file.

    Used by --resume to skip work this *specific model* has already done,
    while still allowing other models to write new pairs to the same file.
    """
    keys: set[tuple[str, str]] = set()
    if not output.exists():
        return keys
    with output.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("passage_id")
            mdl = rec.get("model")
            if pid and mdl:
                keys.add((pid, mdl))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate grounded Q&A pairs from CFM passages via a local Ollama server.",
    )
    ap.add_argument("--passages", type=Path, default=Path("data/passages.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/qa_pairs.jsonl"))
    ap.add_argument(
        "--model",
        default="qwen2.5:3b",
        help="Ollama model name (default: 'qwen2.5:3b').",
    )
    ap.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="Ollama OpenAI-compatible base URL (default: http://localhost:11434/v1).",
    )
    ap.add_argument("--pairs-per-passage", type=int, default=5)
    ap.add_argument(
        "--include-kinds",
        default="ementa,considerando,artigo,item",
        help="Comma-separated passage kinds to include (default: normative ones).",
    )
    ap.add_argument(
        "--keep-boilerplate",
        action="store_true",
        help="Also generate from articles that only approve, put in force or revoke.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Max passages to process.")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--frequency-penalty", type=float, default=0.5)
    ap.add_argument("--presence-penalty", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=None, help="Sampling seed (reproducible runs).")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip passages already present in output and append new ones "
            "(default). Use --no-resume to truncate and start fresh."
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="Do not call the API; emit placeholder pairs.")
    args = ap.parse_args()

    if not args.passages.is_file():
        print(f"Error: passages not found: {args.passages}", file=sys.stderr)
        sys.exit(1)

    include_kinds = {k.strip() for k in args.include_kinds.split(",") if k.strip()}
    passages = _load_passages(args.passages, include_kinds, args.keep_boilerplate)
    if args.limit is not None:
        passages = passages[: args.limit]

    record_model = args.model if not args.dry_run else "dry-run"
    if args.resume:
        existing_keys = _existing_keys(args.output)
        passages = [
            p for p in passages
            if (p.get("passage_id"), record_model) not in existing_keys
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_in = len(passages)
    total_out = 0
    errors = 0
    start = time.time()
    sampling: dict[str, Any] = {
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
    }
    if args.seed is not None:
        sampling["seed"] = args.seed

    def _worker(p: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]] | None, str | None]:
        try:
            pairs = _generate_pairs_for_passage(
                p,
                n_pairs=args.pairs_per_passage,
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                dry_run=args.dry_run,
                sampling=sampling,
            )
            return p, pairs, None
        except Exception as e:
            return p, None, str(e)

    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as out_f:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futures = [ex.submit(_worker, p) for p in passages]
            for i, fut in enumerate(cf.as_completed(futures), start=1):
                p, pairs, err = fut.result()
                pid = p.get("passage_id")
                if err is not None or pairs is None:
                    errors += 1
                    print(f"[err {i}/{total_in}] {pid}: {err}", file=sys.stderr)
                    continue
                model_slug = _model_slug(record_model)
                for j, pair in enumerate(pairs):
                    rec = {
                        "qa_id": f"{pid}__{model_slug}__qa_{j:02d}",
                        "passage_id": pid,
                        "document_id": p.get("document_id"),
                        "kind": p.get("kind"),
                        "label": p.get("label"),
                        "citation": p.get("citation"),
                        "page_start": p.get("page_start"),
                        "page_end": p.get("page_end"),
                        "language": p.get("language", "pt-BR"),
                        "domain": p.get("domain"),
                        "context": p.get("text"),
                        "question": pair["question"],
                        "answer": pair["answer"],
                        "model": record_model,
                        "prompt_version": PROMPT_VERSION,
                        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "review_status": "unreviewed",
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False))
                    out_f.write("\n")
                    total_out += 1
                out_f.flush()
                if i % 10 == 0 or i == total_in:
                    elapsed = time.time() - start
                    print(
                        f"[{i}/{total_in}] pairs_written={total_out} errors={errors} "
                        f"elapsed={elapsed:0.1f}s",
                        file=sys.stderr,
                    )

    print(
        f"\nDone. passages_processed={total_in - errors}/{total_in} "
        f"pairs_written={total_out} errors={errors} "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
