#!/usr/bin/env python3
"""Filter, deduplicate and profile the judged Q&A pairs.

Runs after judge_qa.py, on its --clean-output (pairs that survived the judges):

  1. drop malformed pairs, too-short questions/answers, non-Portuguese text,
     and questions that name the resolution/article/considerando (real users
     do not, and it hands the model the answer's source)
  2. rewrite the answer's trailing "Fonte:" line from the pair's own source
     passage citation (generators often omit or truncate it)
  3. near-duplicate removal: bge-m3 embeddings of question + answer; inside a
     group above --dedupe-threshold the longest answer is kept
  4. contamination check: drop pairs whose question is close to any question
     of the evaluation gold set (--gold)
  5. report with intrinsic quality metrics of the final set, computed as in
     Athaydes et al. (2025): per-pair Shannon entropy, MTLD and FRE-PT
     readability (mean/std/min/max over pairs), density of the top-100 TF-IDF
     terms, token stats, plus corpus-level TTR/MTLD/entropy and, with
     --compare, the same metrics for an older Q&A set and the Jaccard
     vocabulary overlap. The trailing "Fonte:" line is excluded from all
     metrics (identical template on every answer).

  python clean_qa.py --input data/qa_pairs_clean.jsonl \\
    --gold ../evaluation/questions/gold.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ollama_client import embed, looks_portuguese, words

RE_TRAILING_FONTE = re.compile(r"\s*\(?\s*Fonte\s*:[^\n]*\)?\.?\s*$", re.IGNORECASE)
RE_CITED_QUESTION = re.compile(
    r"Resolu[çc][ãa]o\s+(?:CFM|n[º°o])|\bArt(?:igo)?\.?\s*\d+|\bconsiderando\s*\d+", re.IGNORECASE
)
RE_SUSPENDED = re.compile(r"artigo\s+suspenso\s+por\s+decis[ãa]o\s+judicial", re.IGNORECASE)
RE_VOWEL_GROUP = re.compile(r"[aeiouáéíóúâêôãõàü]+", re.IGNORECASE)
RE_SENTENCE_END = re.compile(r"[.!?;:]+")
TOP_TERMS = 100
# Function words left out of the TF-IDF term ranking (domain words such as
# "médico" or "paciente" stay in).
STOPWORDS = {
    "a", "o", "as", "os", "e", "é", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "um", "uma", "uns", "umas", "para", "por", "pelo", "pela", "pelos",
    "pelas", "com", "sem", "que", "se", "ou", "ao", "aos", "à", "às", "seu", "sua",
    "seus", "suas", "ser", "são", "está", "estão", "foi", "ele", "ela", "eles", "elas",
    "isso", "esse", "essa", "esses", "essas", "este", "esta", "estes", "estas", "qual",
    "quais", "quem", "quando", "como", "onde", "mais", "menos", "mas", "não", "sim",
    "já", "também", "até", "entre", "mesmo", "há", "nº", "n", "º", "lhe", "me", "sobre",
    "deve", "devem", "pode", "podem", "caso", "casos", "outro", "outra", "outros",
    "outras", "todo", "toda", "todos", "todas", "cada", "seja", "sejam", "ter", "tem",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix == ".json":
        data = json.loads(text)
        return data if isinstance(data, list) else data.get("questions", [])
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def with_source_line(answer: str, citation: str) -> tuple[str, bool]:
    """Answer with every trailing Fonte line replaced by 'Fonte: <citation>'."""
    body = answer.strip()
    while True:
        stripped = RE_TRAILING_FONTE.sub("", body)
        if stripped == body:
            break
        body = stripped
    fixed = f"{body}\n\nFonte: {citation}"
    return fixed, fixed != answer.strip()


def first_reject_reason(rec: dict[str, Any], min_q: int, min_a: int, allow_cited: bool) -> str | None:
    q, a = (rec.get("question") or "").strip(), (rec.get("answer") or "").strip()
    if not q or not a or not rec.get("passage_id") or not rec.get("citation"):
        return "malformed"
    if rec.get("review_status") == "auto_discarded":
        return "judge_discarded"
    if not allow_cited and RE_CITED_QUESTION.search(q):
        return "question_cites_source"
    if len(words(q)) < min_q or len(words(RE_TRAILING_FONTE.sub("", a))) < min_a:
        return "too_short"
    if not looks_portuguese(f"{q} {a}"):
        return "not_portuguese"
    return None


def embed_all(texts: list[str], base_url: str, model: str, batch: int, timeout: int) -> np.ndarray:
    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch):
        vecs.extend(embed(base_url, model, texts[i : i + batch], timeout))
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", file=sys.stderr)
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def dedupe(records: list[dict[str, Any]], vecs: np.ndarray, threshold: float) -> dict[int, int]:
    """Greedy, longest answer first. Returns {dropped_index: kept_index}."""
    order = sorted(range(len(records)), key=lambda i: -len(records[i]["answer"]))
    sims = vecs @ vecs.T
    kept: list[int] = []
    dropped: dict[int, int] = {}
    for i in order:
        if kept:
            row = sims[i, kept]
            j = int(np.argmax(row))
            if row[j] >= threshold:
                dropped[i] = kept[j]
                continue
        kept.append(i)
    return dropped


def mtld_pass(tokens: list[str], threshold: float = 0.72) -> float:
    factors, types, count = 0.0, set(), 0
    for tok in tokens:
        count += 1
        types.add(tok)
        if len(types) / count <= threshold:
            factors += 1
            types, count = set(), 0
    if count:
        factors += (1 - len(types) / count) / (1 - threshold)
    return len(tokens) / factors if factors else float(len(tokens))


def mtld(tokens: list[str]) -> float:
    return (mtld_pass(tokens) + mtld_pass(tokens[::-1])) / 2


def shannon_entropy(tokens: list[str]) -> float:
    n = len(tokens)
    return -sum((c / n) * math.log2(c / n) for c in collections.Counter(tokens).values()) if n else 0.0


def syllables(word: str) -> int:
    return max(1, len(RE_VOWEL_GROUP.findall(word)))


def fre_pt(text: str) -> float | None:
    """Flesch Reading Ease adapted to Brazilian Portuguese (higher = easier)."""
    toks = words(text)
    if not toks:
        return None
    sentences = max(1, len([s for s in RE_SENTENCE_END.split(text) if words(s)]))
    syl = sum(syllables(t) for t in toks)
    return 248.835 - 1.015 * (len(toks) / sentences) - 84.6 * (syl / len(toks))


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 3),
        "std": round(statistics.pstdev(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def term_density(docs: list[list[str]], top_n: int = TOP_TERMS) -> dict[str, Any]:
    """Share of all words taken by the top-N TF-IDF content terms (pairs as documents)."""
    df: collections.Counter[str] = collections.Counter()
    for doc in docs:
        df.update(set(doc))
    n_docs = len(docs)
    scores: collections.Counter[str] = collections.Counter()
    for doc in docs:
        if not doc:
            continue
        for term, count in collections.Counter(doc).items():
            if term in STOPWORDS or len(term) < 3:
                continue
            scores[term] += (count / len(doc)) * math.log(n_docs / df[term])
    top = [term for term, _ in scores.most_common(top_n)]
    top_set = set(top)
    total = sum(len(doc) for doc in docs)
    hits = sum(1 for doc in docs for tok in doc if tok in top_set)
    return {"top_n": top_n, "density": round(hits / total, 4) if total else None, "top_terms": top[:20]}


def quality_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    pairs = [(r["question"], RE_TRAILING_FONTE.sub("", r["answer"])) for r in records]
    q_lens = [len(words(q)) for q, _ in pairs]
    a_lens = [len(words(a)) for _, a in pairs]
    totals = [q + a for q, a in zip(q_lens, a_lens)]
    docs = [words(f"{q} {a}") for q, a in pairs]
    tokens = [t for doc in docs for t in doc]
    freq = collections.Counter(tokens)
    readability = [s for s in (fre_pt(f"{q} {a}") for q, a in pairs) if s is not None]
    return {
        "pairs": len(records),
        "tokens_per_pair": {
            "mean": round(statistics.mean(totals), 2),
            "std": round(statistics.pstdev(totals), 2),
            "min": min(totals),
            "max": max(totals),
        },
        "tokens_per_question_mean": round(statistics.mean(q_lens), 2),
        "tokens_per_answer_mean": round(statistics.mean(a_lens), 2),
        "per_pair": {
            "shannon_entropy_bits": describe([shannon_entropy(d) for d in docs if d]),
            "mtld": describe([mtld(d) for d in docs if d]),
            "fre_pt": describe(readability) if readability else None,
        },
        "term_density": term_density(docs),
        "corpus": {
            "vocabulary_size": len(freq),
            "ttr": round(len(freq) / len(tokens), 4),
            "mtld": round(mtld(tokens), 2),
            "shannon_entropy_bits": round(shannon_entropy(tokens), 3),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter, dedupe and profile judged CFM Q&A pairs.")
    ap.add_argument("--input", type=Path, default=Path("data/qa_pairs_clean.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/qa_pairs_final.jsonl"))
    ap.add_argument("--rejected", type=Path, default=Path("data/qa_pairs_rejected.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("data/qa_clean_report.json"))
    ap.add_argument("--gold", type=Path, default=None, help="evaluation questions (JSON/JSONL with 'question')")
    ap.add_argument(
        "--passages",
        type=Path,
        default=None,
        help="current passages.jsonl; pairs whose passage_id is no longer in it are dropped",
    )
    ap.add_argument("--compare", type=Path, default=None, help="older Q&A JSONL for the vocabulary comparison")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--dedupe-threshold", type=float, default=0.92)
    ap.add_argument("--contamination-threshold", type=float, default=0.90)
    ap.add_argument("--min-question-words", type=int, default=5)
    ap.add_argument("--min-answer-words", type=int, default=8, help="answer words, not counting the Fonte line")
    ap.add_argument("--allow-cited-questions", action="store_true", help="keep questions that name the resolution/article")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    records = load_records(args.input)
    reasons: collections.Counter[str] = collections.Counter()
    rejected: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
    fonte_fixed = 0
    current = {p["passage_id"]: p.get("text") or "" for p in load_records(args.passages)} if args.passages else None
    for rec in records:
        reason = first_reject_reason(rec, args.min_question_words, args.min_answer_words, args.allow_cited_questions)
        if reason is None and current is not None:
            if rec.get("passage_id") not in current:
                reason = "passage_not_in_corpus"
            elif RE_SUSPENDED.search(current[rec["passage_id"]]):
                reason = "suspended_article"
        if reason:
            reasons[reason] += 1
            rejected.append({**rec, "reject_reason": reason})
            continue
        answer, changed = with_source_line(rec["answer"], rec["citation"])
        fonte_fixed += changed
        survivors.append({**rec, "answer": answer} if changed else rec)
    print(f"{len(records)} pairs, {len(survivors)} after filters ({fonte_fixed} Fonte lines rewritten)", file=sys.stderr)

    if survivors:
        vecs = embed_all(
            [f"{r['question']}\n{r['answer']}" for r in survivors],
            args.base_url, args.embed_model, args.batch_size, args.timeout,
        )
        dropped = dedupe(survivors, vecs, args.dedupe_threshold)
        for i, kept in dropped.items():
            reasons["near_duplicate"] += 1
            rejected.append({**survivors[i], "reject_reason": "near_duplicate", "duplicate_of": survivors[kept]["qa_id"]})
        survivors = [r for i, r in enumerate(survivors) if i not in dropped]
        print(f"{len(survivors)} after dedupe (threshold {args.dedupe_threshold})", file=sys.stderr)

    if args.gold and survivors:
        gold = [g for g in load_records(args.gold) if g.get("question")]
        gvecs = embed_all([g["question"] for g in gold], args.base_url, args.embed_model, args.batch_size, args.timeout)
        qvecs = embed_all([r["question"] for r in survivors], args.base_url, args.embed_model, args.batch_size, args.timeout)
        sims = qvecs @ gvecs.T
        best = sims.argmax(axis=1)
        clean: list[dict[str, Any]] = []
        for i, rec in enumerate(survivors):
            score = float(sims[i, best[i]])
            if score >= args.contamination_threshold:
                reasons["gold_contamination"] += 1
                rejected.append({**rec, "reject_reason": "gold_contamination", "gold_id": gold[best[i]].get("id"), "gold_similarity": round(score, 4)})
            else:
                clean.append(rec)
        survivors = clean
        print(f"{len(survivors)} after contamination check (threshold {args.contamination_threshold})", file=sys.stderr)

    for path, rows in ((args.output, survivors), (args.rejected, rejected)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report: dict[str, Any] = {
        "input": str(args.input),
        "input_pairs": len(records),
        "final_pairs": len(survivors),
        "rejected": dict(reasons),
        "fonte_lines_rewritten": fonte_fixed,
        "final_by_document": dict(collections.Counter(r.get("document_id") for r in survivors)),
        "final_by_kind": dict(collections.Counter(r.get("kind") for r in survivors)),
        "passages_covered": len({r["passage_id"] for r in survivors}),
        "settings": {
            "embed_model": args.embed_model,
            "dedupe_threshold": args.dedupe_threshold,
            "contamination_threshold": args.contamination_threshold if args.gold else None,
            "min_question_words": args.min_question_words,
            "min_answer_words": args.min_answer_words,
            "allow_cited_questions": args.allow_cited_questions,
        },
        "metrics": {"input": quality_metrics(records), "final": quality_metrics(survivors)},
    }
    if args.compare:
        old = [r for r in load_records(args.compare) if r.get("question") and r.get("answer")]
        report["metrics"]["compare"] = quality_metrics(old)
        def vocab(rows: list[dict[str, Any]]) -> set[str]:
            return {t for r in rows for t in words(f"{r['question']} {RE_TRAILING_FONTE.sub('', r['answer'])}")}

        vocab_old, vocab_new = vocab(old), vocab(survivors)
        union = vocab_old | vocab_new
        report["vocabulary_jaccard_vs_compare"] = round(len(vocab_old & vocab_new) / len(union), 4) if union else None
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("input_pairs", "final_pairs", "rejected", "fonte_lines_rewritten")}, ensure_ascii=False))
    print(f"-> {args.output}\n-> {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
