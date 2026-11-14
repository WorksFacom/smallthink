#!/usr/bin/env python3
"""Pair model answers with the golden set into Gemini judge items.

Writes <out_dir>/<stem>.json (all items) and <out_dir>/<stem>/batch_NNN.json
(chunks to paste one request at a time). Model identity is dropped and
reasoning blocks are stripped so the judge sees only the final answer.

  python build_eval_set.py answers/finetuned-rag.jsonl
  python build_eval_set.py answers/gpt-oss-120b.jsonl --no-sources
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

GOLD_KEYS = (
    "question_type",
    "answerable",
    "short_answer",
    "reference",
    "key_points",
    "gold_citation",
    "must_not",
    "notes",
)
REASONING_RE = re.compile(r"<(think|thought)>.*?</\1>", re.IGNORECASE | re.DOTALL)
OUTPUT_RE = re.compile(r"<output>(.*?)</output>", re.IGNORECASE | re.DOTALL)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def final_answer(text: str) -> str:
    text = REASONING_RE.sub("", text)
    match = OUTPUT_RE.search(text)
    return (match.group(1) if match else text).strip()


def judge_item(gold: dict[str, Any], row: dict[str, Any], with_sources: bool) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": gold["id"],
        "question": gold["question"],
        "gold": {key: gold.get(key) for key in GOLD_KEYS},
        "answer": final_answer(row.get("answer") or ""),
    }
    item["gold"]["evidence"] = [
        {"passage_id": ev.get("passage_id"), "quote": ev.get("quote")}
        for ev in gold.get("evidence") or []
    ]
    if with_sources:
        item["sources"] = [
            {"citation": src.get("citation"), "text": src.get("text")}
            for src in row.get("sources") or []
        ]
    return item


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gemini judge items from answers + golden set.")
    parser.add_argument("answers", type=Path, help="answers JSONL written by compare.py")
    parser.add_argument("--gold", type=Path, default=Path("questions/gold.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("judge_sets"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="omit retrieved passages (closed-book runs)",
    )
    args = parser.parse_args()

    gold_rows = load_jsonl(args.gold)
    answers = {str(row["id"]): row for row in load_jsonl(args.answers) if not row.get("error")}

    items: list[dict[str, Any]] = []
    missing: list[str] = []
    for gold in gold_rows:
        row = answers.get(str(gold["id"]))
        if row is None or not (row.get("answer") or "").strip():
            missing.append(str(gold["id"]))
            continue
        items.append(judge_item(gold, row, with_sources=not args.no_sources))

    stem = args.answers.stem
    write_json(args.out_dir / f"{stem}.json", items)
    size = max(1, args.batch_size)
    for start in range(0, len(items), size):
        write_json(args.out_dir / stem / f"batch_{start // size + 1:03d}.json", items[start : start + size])

    print(f"wrote {len(items)} items to {args.out_dir / f'{stem}.json'}", file=sys.stderr)
    print(f"wrote {-(-len(items) // size)} batches to {args.out_dir / stem}/", file=sys.stderr)
    if missing:
        print(f"no usable answer for {len(missing)} ids: {', '.join(missing)}", file=sys.stderr)


if __name__ == "__main__":
    main()
