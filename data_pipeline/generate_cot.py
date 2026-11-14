#!/usr/bin/env python3
"""Write Portuguese chain-of-thought for the RAG problems with a local SLM.

The answer is already known (the judged Q&A answer, or the refusal), so the
model only writes the reasoning that leads to it: which passage is relevant,
what it says, conditions/exceptions, conclusion. A Thought is accepted only if
it is long enough, in Portuguese, free of stray markup, and (for answer
problems) names the gold passage's article/considerando/item.

Output rows are Axolotl ``type: alpaca``:
  instruction = problem prompt (identical to the bot's prompt)
  output      = "<Thought>\\n...\\n</Thought>\\n<Output>\\n<solution>\\n</Output>"

Every attempt is logged to --raw, so the run can be resumed; train/val files
are rebuilt from the log at the end. The split is by qa_id, so the answer and
refusal variants of one Q&A land in the same split.

  python generate_cot.py --problems data/rag_problems.jsonl --model qwen3.5:9b
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from ollama_client import chat, extract_json, looks_portuguese

PROMPT_VERSION = "cfm-cot-v1"
FORBIDDEN_MARKUP = re.compile(r"\\boxed|</?(?:think|thinking|thought|output|answer|double-check)>", re.IGNORECASE)

SYSTEM_PROMPT = (
    "Você escreve o raciocínio passo a passo que leva a uma resposta já conhecida, "
    "para treinar um assistente de ética médica e direito médico brasileiro "
    "(normas do Conselho Federal de Medicina).\n\n"
    "Regras:\n"
    "1. Escreva em português do Brasil, em primeira pessoa, de 4 a 8 frases.\n"
    "2. Use SOMENTE os trechos normativos fornecidos. Não invente normas nem artigos.\n"
    "3. Siga esta ordem: (a) o que a pergunta pede; (b) quais trechos são relevantes "
    "e quais não são, identificando-os pelo artigo, considerando ou item (ex.: "
    "\"Art. 73\", \"considerando 8\"); (c) o que o trecho relevante determina, "
    "citando a frase-chave entre aspas; (d) condições ou exceções, se houver; "
    "(e) uma conclusão curta que leva à resposta conhecida.\n"
    "4. Se a resposta conhecida disser que não é possível responder, explique por "
    "que nenhum trecho fornecido contém a informação pedida.\n"
    "5. Não copie a resposta final inteira e não use marcações como <Thought>, "
    "<Output> ou \\boxed.\n"
    "6. Responda SOMENTE com JSON válido: {\"raciocinio\": \"...\"}"
)

USER_TEMPLATE = "{problem}\n\nResposta conhecida:\n{solution}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_pattern(citation: str) -> re.Pattern[str] | None:
    """Regex for the gold passage's own label inside a Thought."""
    tail = (citation or "").split(",", 1)[1].strip() if "," in (citation or "") else ""
    m = re.match(r"Art\.\s*(\d+)", tail)
    if m:
        # (?!\d) instead of \b: in "Art. 1º" the "º" counts as a word character
        return re.compile(rf"\bart(?:igo)?\.?\s*{m.group(1)}(?!\d)", re.IGNORECASE)
    m = re.match(r"considerando\s*(\d+)", tail, re.IGNORECASE)
    if m:
        return re.compile(rf"\bconsiderando\s*{m.group(1)}(?!\d)", re.IGNORECASE)
    m = re.match(r"(?:Cap[íi]tulo\s+[IVXLC]+|Pre[âa]mbulo),\s*([IVXLC]+)$", tail)
    if m:
        return re.compile(rf"\b{m.group(1)}\b")
    if "ementa" in (citation or "").lower():
        return re.compile(r"\bementa\b", re.IGNORECASE)
    return None


def check_thought(thought: str, problem: dict[str, Any], gold_citation: str | None, min_chars: int) -> str | None:
    if len(thought) < min_chars:
        return "too_short"
    if FORBIDDEN_MARKUP.search(thought):
        return "markup"
    if not looks_portuguese(thought):
        return "not_portuguese"
    if problem["kind"] == "answer" and gold_citation:
        pattern = label_pattern(gold_citation)
        if pattern and not pattern.search(thought):
            return "gold_not_mentioned"
    if problem["kind"] != "answer" and "não" not in thought.lower():
        return "refusal_not_explained"
    return None


def generate_one(problem: dict[str, Any], gold_citation: str | None, args: argparse.Namespace) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(problem=problem["problem"], solution=problem["solution"])},
    ]
    reason, thought = "no_attempt", ""
    for attempt in range(1, args.max_attempts + 1):
        try:
            raw = chat(
                args.base_url, args.model, messages,
                temperature=args.temperature, max_tokens=args.max_tokens,
                timeout=args.timeout, json_mode=True,
            )
            obj = json.loads(extract_json(raw))
            thought = re.sub(r"\s+\n", "\n", str(obj.get("raciocinio") or "")).strip()
            reason = check_thought(thought, problem, gold_citation, args.min_thought_chars)
        except Exception as e:  # network/JSON errors count as a failed attempt
            reason = f"error: {type(e).__name__}: {e}"[:200]
        if reason is None:
            return {"status": "accepted", "thought": thought, "attempts": attempt}
    return {"status": "rejected", "reason": reason, "thought": thought, "attempts": args.max_attempts}


def alpaca_row(problem: dict[str, Any], thought: str) -> dict[str, str]:
    return {
        "instruction": problem["problem"],
        "input": "",
        "output": f"<Thought>\n{thought}\n</Thought>\n<Output>\n{problem['solution']}\n</Output>",
    }


def in_val(qa_id: str, ratio: float) -> bool:
    bucket = int(hashlib.sha1(qa_id.encode("utf-8")).hexdigest(), 16) % 1000
    return bucket < ratio * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Portuguese CoT for RAG problems with a local SLM.")
    ap.add_argument("--problems", type=Path, default=Path("data/rag_problems.jsonl"))
    ap.add_argument("--passages", type=Path, default=Path("data/passages.jsonl"))
    ap.add_argument("--raw", type=Path, default=Path("data/cot_raw.jsonl"), help="attempt log (resume)")
    ap.add_argument("--out-prefix", type=Path, default=Path("data/cot_alpaca"))
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--min-thought-chars", type=int, default=200)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    problems = load_jsonl(args.problems)
    citations = {p["passage_id"]: p.get("citation") for p in load_jsonl(args.passages)}
    done = {r["id"] for r in load_jsonl(args.raw)}
    todo = [p for p in problems if p["id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(problems)} problems, {len(done)} already logged, {len(todo)} to generate", file=sys.stderr)

    args.raw.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with args.raw.open("a", encoding="utf-8") as log, cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = {
            ex.submit(generate_one, p, citations.get(p.get("gold_passage_id") or ""), args): p
            for p in todo
        }
        for n, fut in enumerate(cf.as_completed(futures), start=1):
            p = futures[fut]
            result = fut.result()
            log.write(json.dumps({
                "id": p["id"], "qa_id": p["qa_id"], "kind": p["kind"], "model": args.model,
                "prompt_version": PROMPT_VERSION, **result,
            }, ensure_ascii=False) + "\n")
            log.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] elapsed={time.time() - start:0.0f}s", file=sys.stderr)

    by_id = {p["id"]: p for p in problems}
    latest = {r["id"]: r for r in load_jsonl(args.raw)}
    train, val = [], []
    stats: collections.Counter[str] = collections.Counter()
    for pid, rec in latest.items():
        prob = by_id.get(pid)
        if prob is None:
            continue
        if rec["status"] != "accepted":
            stats[f"rejected:{str(rec.get('reason')).split(':')[0]}"] += 1
            continue
        stats[f"accepted:{prob['kind']}"] += 1
        (val if in_val(prob["qa_id"], args.val_ratio) else train).append(alpaca_row(prob, rec["thought"]))

    for suffix, rows in (("_train.jsonl", train), ("_val.jsonl", val)):
        path = Path(f"{args.out_prefix}{suffix}")
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"problems": len(problems), "train_rows": len(train), "val_rows": len(val), "counts": dict(stats), "model": args.model}
    Path(f"{args.out_prefix}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
