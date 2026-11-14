#!/usr/bin/env python3
"""Turn the final Q&A pairs into RAG training problems.

Each problem is the exact prompt the bot builds at inference
(rag_bot/app/rag.py ``build_prompt``): fixed header, "Trechos
normativos" blocks ``[citation]\\ntext``, then ``Pergunta:``.

  answer           gold passage + hard negatives: the passages the bot's own
                   retriever (bge-m3 + Qdrant ``cfm_passages``) ranks highest
                   for the question, excluding other parts of the gold article.
                   Gold position is shuffled.
  refusal_unrelated  only passages from other resolutions, outside the
                   question's top retrieval hits; solution = refusal.
  refusal_empty    no passages ("(nenhum trecho recuperado)"); solution = refusal.

Requires the passages indexed in Qdrant (rag_bot index step).

  python build_rag_problems.py --qa data/qa_pairs_final.jsonl \\
    --passages data/passages.jsonl --qdrant-url http://localhost:6333
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from ollama_client import embed, qdrant_query

# Must stay identical to HEADER / NO_SOURCES_ANSWER / build_prompt() in
# rag_bot/app/rag.py, or training and inference prompts diverge.
HEADER = (
    "Voce e um assistente especializado em etica medica e direito medico brasileiro, "
    "com base nas normas do Conselho Federal de Medicina (CFM). Responda a pergunta "
    "usando exclusivamente os trechos normativos fornecidos, de forma objetiva, e cite "
    "ao final a fonte (resolucao e artigo) que fundamenta a resposta. Se os trechos "
    "fornecidos nao contiverem a informacao necessaria, diga que nao e possivel "
    "responder com base neles."
)
REFUSAL = "Não é possível responder a essa pergunta com base nos trechos normativos fornecidos."
EMPTY_BLOCK = "(nenhum trecho recuperado)"
INDEXED_KINDS = {"ementa", "considerando", "artigo", "item"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_prompt(question: str, passages: list[dict[str, Any]]) -> str:
    if passages:
        chunks = "\n\n".join(f"[{p['citation']}]\n{p['text']}" for p in passages)
    else:
        chunks = EMPTY_BLOCK
    return f"{HEADER}\n\nTrechos normativos:\n\n{chunks}\n\nPergunta: {question.strip()}"


def article_base(passage_id: str) -> str:
    return re.sub(r"_p\d+$", "", passage_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build RAG-format training problems from Q&A pairs.")
    ap.add_argument("--qa", type=Path, default=Path("data/qa_pairs_final.jsonl"))
    ap.add_argument("--passages", type=Path, default=Path("data/passages.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/rag_problems.jsonl"))
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--qdrant-api-key", default=None)
    ap.add_argument("--collection", default="cfm_passages")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--passages-per-problem", default="2,3", help="choices for passages per answer problem")
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--refusal-ratio", type=float, default=0.12, help="extra refusal_unrelated problems, share of Q&As")
    ap.add_argument("--empty-ratio", type=float, default=0.04, help="extra refusal_empty problems, share of Q&As")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sizes = [int(s) for s in args.passages_per_problem.split(",") if s.strip()]
    passages = {p["passage_id"]: p for p in load_jsonl(args.passages)}
    pool = [p for p in passages.values() if p.get("kind") in INDEXED_KINDS and not p.get("duplicate_of")]
    qas = load_jsonl(args.qa)

    problems: list[dict[str, Any]] = []
    retrieved_by_qa: dict[str, list[str]] = {}
    counts: collections.Counter[str] = collections.Counter()

    for i, qa in enumerate(qas, start=1):
        gold_id = qa.get("passage_id")
        gold = passages.get(gold_id or "")
        if gold is None:
            counts["skipped_gold_not_in_passages"] += 1
            continue
        if gold.get("duplicate_of"):
            gold = passages[gold["duplicate_of"]]
        try:
            vec = embed(args.base_url, args.embed_model, [qa["question"]], args.timeout)[0]
            hits = qdrant_query(args.qdrant_url, args.collection, vec, args.retrieve_k, args.timeout, args.qdrant_api_key)
        except Exception as e:  # keep going; fall back to same-document passages
            print(f"[warn] retrieval failed for {qa.get('qa_id')}: {e}", file=sys.stderr)
            hits = []
        retrieved = [h["passage_id"] for h in hits if h.get("passage_id") in passages]
        retrieved_by_qa[qa["qa_id"]] = retrieved

        n_total = rng.choice(sizes)
        gold_base = article_base(gold["passage_id"])
        negatives = [pid for pid in retrieved if article_base(pid) != gold_base][: n_total - 1]
        if len(negatives) < n_total - 1:
            same_doc = [
                p["passage_id"] for p in pool
                if p["document_id"] == gold["document_id"]
                and article_base(p["passage_id"]) != gold_base
                and p["passage_id"] not in negatives
            ]
            negatives += rng.sample(same_doc, min(len(same_doc), n_total - 1 - len(negatives)))
            counts["answer_with_fallback_negatives"] += 1
        blocks = [gold] + [passages[pid] for pid in negatives]
        rng.shuffle(blocks)
        problems.append(
            {
                "kind": "answer",
                "qa_id": qa["qa_id"],
                "question": qa["question"],
                "gold_passage_id": gold["passage_id"],
                "passage_ids": [p["passage_id"] for p in blocks],
                "problem": build_prompt(qa["question"], blocks),
                "solution": qa["answer"].strip(),
            }
        )
        counts["answer"] += 1
        if i % 50 == 0:
            print(f"[{i}/{len(qas)}] problems={len(problems)}", file=sys.stderr)

    answered = [p for p in problems if p["kind"] == "answer"]
    by_qa = {qa["qa_id"]: qa for qa in qas}
    for source in rng.sample(answered, min(len(answered), round(len(qas) * args.refusal_ratio))):
        gold_doc = passages[source["gold_passage_id"]]["document_id"]
        avoid = set(retrieved_by_qa.get(source["qa_id"], []))
        candidates = [p for p in pool if p["document_id"] != gold_doc and p["passage_id"] not in avoid]
        blocks = rng.sample(candidates, min(len(candidates), rng.choice(sizes)))
        question = by_qa[source["qa_id"]]["question"]
        problems.append(
            {
                "kind": "refusal_unrelated",
                "qa_id": source["qa_id"],
                "question": question,
                "gold_passage_id": None,
                "passage_ids": [p["passage_id"] for p in blocks],
                "problem": build_prompt(question, blocks),
                "solution": REFUSAL,
            }
        )
        counts["refusal_unrelated"] += 1

    for source in rng.sample(answered, min(len(answered), round(len(qas) * args.empty_ratio))):
        question = by_qa[source["qa_id"]]["question"]
        problems.append(
            {
                "kind": "refusal_empty",
                "qa_id": source["qa_id"],
                "question": question,
                "gold_passage_id": None,
                "passage_ids": [],
                "problem": build_prompt(question, []),
                "solution": REFUSAL,
            }
        )
        counts["refusal_empty"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for n, prob in enumerate(problems):
            f.write(json.dumps({"id": f"cfm_{n:05d}", **prob}, ensure_ascii=False) + "\n")
    print(json.dumps(dict(counts), ensure_ascii=False))
    print(f"-> {args.output} ({len(problems)} problems)", file=sys.stderr)


if __name__ == "__main__":
    main()
