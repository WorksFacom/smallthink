#!/usr/bin/env python3
"""BERTScore of each system's answers against the gold references.

Secondary, judge-independent metric. Before scoring, reasoning blocks, leftover
tags, markdown and "Fonte:" lines are removed from both sides, so the score
compares content rather than formatting or the citation template.

  answer vs reference   precision / recall / F1 per answer
  answer vs key points  (--key-points) recall of each gold key point inside the
                        answer, averaged per answer

Reported overall, per answerability and per question type; with --baseline,
paired bootstrap differences against that system; when judge output exists in
--judgments-dir, Spearman correlation between F1 and the judge's correctness.
BERTScore cannot see negation or citation errors, so read it next to the judge.

  pip install -r requirements-bertscore.txt
  python bertscore_eval.py                                   # every answers/*.jsonl
  python bertscore_eval.py --systems finetuned-rag qwen3-4b-rag gpt-oss-120b-rag \\
    --baseline finetuned-rag --key-points

Writes results/bertscore.json (summary) and results/bertscore_items.jsonl
(one row per scored answer) and prints a summary table.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
import sys
import warnings
from pathlib import Path
from typing import Any

REASONING_RE = re.compile(r"<(think|thinking|thought)>.*?</\1>", re.S | re.I)
TAG_RE = re.compile(r"</?(?:think|thinking|thought|output|answer)>", re.I)
SOURCE_LINE_RE = re.compile(r"(?im)^\s*\(?\s*fontes?\s*:.*$")
TRAILING_SOURCE_RE = re.compile(r"\s*\(?\s*fontes?\s*:[^\n]*$", re.I)
MARKDOWN_RULES = [
    (re.compile(r"\*\*|__|`"), ""),
    (re.compile(r"(?m)^\s{0,3}#{1,6}\s*"), ""),
    (re.compile(r"(?m)^\s*[-*•]\s+"), ""),
    (re.compile(r"(?m)^\s*\d+[.)]\s+"), ""),
    (re.compile(r"\|"), " "),
    (re.compile(r"(?m)^[\s:\-]+$"), ""),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean(text: str, keep_source: bool) -> str:
    text = REASONING_RE.sub(" ", text or "")
    text = TAG_RE.sub(" ", text).replace("\u202f", " ").replace("\u00a0", " ")
    if not keep_source:
        text = TRAILING_SOURCE_RE.sub("", SOURCE_LINE_RE.sub("", text))
    for pattern, repl in MARKDOWN_RULES:
        text = pattern.sub(repl, text)
    return re.sub(r"\s+", " ", text).strip()


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []


def final_answers(path: Path) -> dict[str, str]:
    """Last successful, non-empty answer per id (reruns append rows)."""
    out: dict[str, str] = {}
    for row in load_jsonl(path):
        if row.get("error") or not (row.get("answer") or "").strip():
            continue
        out[str(row["id"])] = row["answer"]
    return out


def cap_max_length(scorer: Any) -> Any:
    """Tokenizers without model_max_length (e.g. BERTimbau, Legal-BERTimbau) report
    ~1e30, which bert-score passes as the truncation length and the fast tokenizer
    cannot convert; cap it at the model's position limit."""
    tokenizer = scorer._tokenizer
    limit = getattr(scorer._model.config, "max_position_embeddings", 512)
    if tokenizer.model_max_length > limit:
        tokenizer.model_max_length = limit
    return scorer


def make_scorer(args: argparse.Namespace) -> tuple[Any, bool]:
    from bert_score import BERTScorer

    kwargs: dict[str, Any] = {"model_type": args.model, "lang": "pt", "batch_size": args.batch_size}
    if args.num_layers is not None:
        kwargs["num_layers"] = args.num_layers
    if args.device:
        kwargs["device"] = args.device
    if args.rescale:
        scorer = cap_max_length(BERTScorer(rescale_with_baseline=True, **kwargs))
        try:
            scorer.baseline_vals  # noqa: B018 - raises when no baseline file exists
            return scorer, True
        except (ValueError, FileNotFoundError, OSError) as e:
            warnings.warn(f"no rescaling baseline for {args.model} / pt ({e}); reporting raw scores")
    return cap_max_length(BERTScorer(rescale_with_baseline=False, **kwargs)), False


def score_pairs(scorer: Any, cands: list[str], refs: list[str]) -> tuple[list[float], list[float], list[float]]:
    if not cands:
        return [], [], []
    p, r, f = scorer.score(cands, refs, verbose=False)
    return p.tolist(), r.tolist(), f.tolist()


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "precision": mean([r["precision"] for r in rows]),
        "recall": mean([r["recall"] for r in rows]),
        "f1": mean([r["f1"] for r in rows]),
        "key_point_recall": mean([r.get("key_point_recall") for r in rows]),
    }


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        groups[str(r.get(key))].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n: int, seed: int) -> dict[str, Any] | None:
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    diffs = [a[i] - b[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return {
        "n": len(ids),
        "mean_diff": round(statistics.fmean(diffs), 4),
        "ci95": [round(boots[int(0.025 * n)], 4), round(boots[int(0.975 * n) - 1], 4)],
        "wins": sum(d > 0 for d in diffs),
        "losses": sum(d < 0 for d in diffs),
    }


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 4) if den else None


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="BERTScore of system answers against the gold references.")
    ap.add_argument("--gold", type=Path, default=Path("questions/gold.jsonl"))
    ap.add_argument("--answers-dir", type=Path, default=Path("answers"))
    ap.add_argument("--judgments-dir", type=Path, default=Path("judgments"))
    ap.add_argument("--systems", nargs="*", help="answer file stems (default: every *.jsonl in --answers-dir)")
    ap.add_argument("--model", default="xlm-roberta-large", help="BERTScore backbone (multilingual)")
    ap.add_argument("--num-layers", type=int, default=None, help="layer to use (default: bert-score's choice for the model)")
    ap.add_argument("--rescale", action="store_true", help="rescale with bert-score's baseline when one exists for the model")
    ap.add_argument("--key-points", action="store_true", help="also score each gold key point against the answer")
    ap.add_argument("--keep-source", action="store_true", help="keep 'Fonte:' lines when scoring")
    ap.add_argument("--baseline", default=None, help="system to compare the others against (paired bootstrap)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--summary", type=Path, default=Path("results/bertscore.json"))
    ap.add_argument("--items", type=Path, default=Path("results/bertscore_items.jsonl"))
    args = ap.parse_args()

    gold = {str(g["id"]): g for g in load_jsonl(args.gold)}
    systems = args.systems or sorted(p.stem for p in args.answers_dir.glob("*.jsonl"))
    scorer, rescaled = make_scorer(args)

    items: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "settings": {
            "gold": str(args.gold),
            "model": args.model,
            "num_layers": args.num_layers,
            "rescaled": rescaled,
            "source_lines_removed": not args.keep_source,
        },
        "systems": {},
    }
    for system in systems:
        answers = final_answers(args.answers_dir / f"{system}.jsonl")
        ids = [i for i in gold if i in answers]
        if not ids:
            print(f"[skip] {system}: no answers matching the gold ids", file=sys.stderr)
            continue
        cands = [clean(answers[i], args.keep_source) for i in ids]
        refs = [clean(gold[i].get("reference") or "", args.keep_source) for i in ids]
        print(f"{system}: scoring {len(ids)} answers", file=sys.stderr)
        p, r, f = score_pairs(scorer, cands, refs)
        rows = [
            {
                "system": system,
                "id": i,
                "answerable": gold[i].get("answerable"),
                "question_type": gold[i].get("question_type"),
                "precision": round(p[k], 4),
                "recall": round(r[k], 4),
                "f1": round(f[k], 4),
            }
            for k, i in enumerate(ids)
        ]

        if args.key_points:
            kp_index = [(k, kp) for k, i in enumerate(ids) for kp in as_list(gold[i].get("key_points"))]
            _, kp_r, _ = score_pairs(
                scorer, [cands[k] for k, _ in kp_index], [clean(kp, args.keep_source) for _, kp in kp_index]
            )
            per_item: dict[int, list[float]] = collections.defaultdict(list)
            for (k, _), value in zip(kp_index, kp_r):
                per_item[k].append(value)
            for k, row in enumerate(rows):
                row["key_point_recall"] = round(statistics.fmean(per_item[k]), 4) if per_item.get(k) else None

        judged = {str(j["id"]): j for j in load_jsonl(args.judgments_dir / f"{system}.jsonl")}
        pairs = [(row["f1"], judged[row["id"]]["correctness"]) for row in rows
                 if row["id"] in judged and judged[row["id"]].get("correctness") is not None]
        items.extend(rows)
        summary["systems"][system] = {
            "answers": len(answers),
            "scored": len(rows),
            "missing_gold_ids": len(gold) - len(ids),
            "overall": summarize(rows),
            "by_answerable": grouped(rows, "answerable"),
            "by_question_type": grouped(rows, "question_type"),
            "spearman_f1_vs_judge_correctness": spearman([a for a, _ in pairs], [float(b) for _, b in pairs]),
            "judged_pairs": len(pairs),
        }

    if args.baseline and args.baseline in summary["systems"]:
        by_system: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
        for row in items:
            by_system[row["system"]][row["id"]] = row
        base = by_system[args.baseline]
        comparisons = {}
        for system, rows in by_system.items():
            if system == args.baseline:
                continue
            comp = {"f1": paired_bootstrap(
                {i: base[i]["f1"] for i in base}, {i: rows[i]["f1"] for i in rows}, args.bootstrap, args.seed)}
            if args.key_points:
                comp["key_point_recall"] = paired_bootstrap(
                    {i: base[i]["key_point_recall"] for i in base if base[i].get("key_point_recall") is not None},
                    {i: rows[i]["key_point_recall"] for i in rows if rows[i].get("key_point_recall") is not None},
                    args.bootstrap, args.seed,
                )
            comparisons[f"{args.baseline} - {system}"] = comp
        summary["paired_vs_baseline"] = comparisons

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.items.open("w", encoding="utf-8") as f:
        for row in items:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nBERTScore ({args.model}, {'rescaled' if rescaled else 'raw'})")
    print("| system | n | P | R | F1 | key-point R | F1 answerable | F1 unanswerable | Spearman F1~judge |")
    print("|---|---|---|---|---|---|---|---|---|")
    for system, s in summary["systems"].items():
        o, ba = s["overall"], s["by_answerable"]
        print(
            f"| {system} | {o['n']} | {fmt(o['precision'])} | {fmt(o['recall'])} | {fmt(o['f1'])} | "
            f"{fmt(o['key_point_recall'])} | {fmt(ba.get('yes', {}).get('f1'))} | {fmt(ba.get('no', {}).get('f1'))} | "
            f"{fmt(s['spearman_f1_vs_judge_correctness'])} |"
        )
    for name, comp in summary.get("paired_vs_baseline", {}).items():
        parts = [f"{metric}: {c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] (n={c['n']})"
                 for metric, c in comp.items() if c]
        print(f"{name}: " + "; ".join(parts))
    print(f"\n-> {args.summary}\n-> {args.items}", file=sys.stderr)


if __name__ == "__main__":
    main()
