#!/usr/bin/env python3
"""Gemini rubric judge for the items written by build_eval_set.py.

One request per item: the rubric is the system instruction (identical in every
request, so it can be billed at the cached-input rate) and the item JSON is the
user message. The response schema fixes the keys and the length of key_points;
judgments that break the rubric's own rules (abstention vs correctness, null
fields, a "2" with missing key points, ...) are judged again with the errors.

  export GEMINI_API_KEY=...
  python judge.py judge_sets/finetuned-rag.json --limit 5 --model gemini-3.8-flash   # trial
  python judge.py judge_sets/*.json --batch

Writes judgments/<stem>.jsonl, one judgment per item (plus judge_model and,
when the retries ran out, judge_errors). A rerun skips judged ids.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from build_gold import batch_request, call_json, estimated_tokens, load_records, split_by_tokens, submit_batch, wait_batch

JUDGE_SYSTEM = """You are an impartial evaluator of answers about Brazilian medical ethics and medical law (normas do Conselho Federal de Medicina, CFM). You compare a MODEL ANSWER against a GOLD RECORD built from the CFM resolutions.

## Input
One JSON object with:
- "id", "question"
- "gold": the gold record, with fields
    "question_type" (binary | factual | list | explanatory),
    "answerable" (yes | partial | no),
    "short_answer" (expected conclusion, or null),
    "reference" (reference answer),
    "key_points" (statements a correct answer must contain),
    "gold_citation" (expected legal source; empty when answerable is "no"),
    "evidence" (verbatim quotes from the normative passages, each with a passage_id),
    "must_not" (claims or citations a correct answer must not make),
    "notes"
- "answer": the model's final answer (reasoning already removed)
- "sources": the normative passages the model received as context (missing for closed-book models, possibly empty)

## Hard rules
1. Judge ONLY against the gold record and the provided passages ("evidence" and "sources"). Do NOT use your own knowledge of CFM resolutions or Brazilian law, even if you believe the gold record is incomplete.
2. Ignore style, formatting, length, tone and markdown. A short correct answer is as good as a long correct answer.
3. Extra information is acceptable only if it does not contradict the gold record or the passages. Extra information that contradicts them counts as a contradiction.
4. A key point is "covered" if the answer states its meaning, even with different words. It is "partial" if only part of it is stated, or it is stated without a required condition or exception. Otherwise it is "missing".
5. A citation is any reference to a resolution, article, inciso, paragraph or "considerando".
   - It "matches gold" if it refers to the same resolution AND the same article (or considerando) as "gold_citation".
   - It is "supported" if the cited text appears in "evidence" or "sources" and actually supports the claim it is attached to.
   - It is "unverifiable" if the cited resolution/article does not appear in "evidence" or "sources". Do not decide on your own whether it exists.
   - It is "misattributed" if the cited resolution/article does appear in "evidence" or "sources" but its text does not say what the answer claims.
6. Abstention ("não é possível responder com base nos trechos", or similar) and answerability:
   - answerable = "yes": abstaining is an error, so "correctness" is 0.
   - answerable = "no": abstaining is the correct behavior. Some of these questions ask for a detail the resolutions do not set (a number, a deadline, a format, a test score) about a topic they do cover; an answer that supplies such a detail, or presents a related rule as if it settled the question, is answering as if the information existed.
   - answerable = "partial": the ideal answer gives the part supported by the evidence and states what cannot be answered.
7. Write "justification" in Brazilian Portuguese, one or two sentences.

## Scoring
- "correctness":
    2 = all key points covered, no contradiction, correct conclusion
    1 = some key points covered or partial, or correct conclusion with important omissions
    0 = wrong conclusion, contradicts the gold record, misses all key points, or abstains on an answerable question
  For answerable = "no": 2 if it abstains without inventing rules; 1 if it abstains but also asserts unsupported rules; 0 if it answers as if the information existed.
- "short_answer_match": true or false when gold "short_answer" is not null (false if the answer abstains); null otherwise. When gold is "Depende", an answer that opens with "Sim" or "Não" but states the same condition or exception under which the outcome changes also matches; one that states a single outcome without that condition does not.
- "faithfulness" (only when "sources" is non-empty; otherwise null):
    2 = every substantive claim is supported by "sources" or "evidence"
    1 = some claims go beyond the passages without contradicting them
    0 = claims contradict the passages, or the answer relies mainly on content not in them

## Output fields
- "key_points": one label per gold key point, same order ([] if gold has none)
- "abstained": whether the answer abstains (declines to answer) instead of answering
- "cites_source": whether the answer cites any resolution, article or considerando; "citations" lists each one with its status
- "citation_matches_gold": "full" = every gold citation matched; "partial" = at least one; "none" = no gold citation matched (including when the answer cites nothing); "not_applicable" = gold_citation is empty
- "must_not_violated": the gold must_not items the answer violates
- "language_pt": the answer is in Portuguese
- "foreign_law": the answer invokes HIPAA, GDPR or any non-Brazilian law"""

KEYS = (
    "id", "correctness", "key_points", "short_answer_match", "contradiction", "abstained",
    "cites_source", "citation_matches_gold", "citations", "must_not_violated", "faithfulness", "language_pt",
    "foreign_law", "justification",
)
STRING = {"type": "STRING"}
BOOLEAN = {"type": "BOOLEAN"}


def judge_schema(item: dict[str, Any]) -> dict[str, Any]:
    n_points = len(item["gold"].get("key_points") or [])
    must_not = [m for m in item["gold"].get("must_not") or [] if m]
    violated: dict[str, Any] = {"type": "ARRAY", "items": {"type": "STRING", "enum": must_not} if must_not else STRING}
    if not must_not:
        violated["maxItems"] = 0
    return {
        "type": "OBJECT",
        "properties": {
            "id": STRING,
            "correctness": {"type": "INTEGER"},
            "key_points": {
                "type": "ARRAY",
                "items": {"type": "STRING", "enum": ["covered", "partial", "missing"]},
                "minItems": n_points,
                "maxItems": n_points,
            },
            "short_answer_match": {"type": "BOOLEAN", "nullable": True},
            "contradiction": BOOLEAN,
            "abstained": BOOLEAN,
            "cites_source": BOOLEAN,
            "citation_matches_gold": {"type": "STRING", "enum": ["full", "partial", "none", "not_applicable"]},
            "citations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {"text": STRING, "status": {"type": "STRING", "enum": ["supported", "unverifiable", "misattributed"]}},
                    "required": ["text", "status"],
                },
            },
            "must_not_violated": violated,
            "faithfulness": {"type": "INTEGER", "nullable": True},
            "language_pt": BOOLEAN,
            "foreign_law": BOOLEAN,
            "justification": STRING,
        },
        "required": list(KEYS),
        "propertyOrdering": list(KEYS),
    }


def user_prompt(item: dict[str, Any], errors: list[str] | None) -> str:
    text = "ITEM:\n" + json.dumps(item, ensure_ascii=False, indent=1)
    if errors:
        text += "\n\nYOUR PREVIOUS JUDGMENT OF THIS ITEM BROKE THESE RULES. Judge it again and fix them:\n" + "\n".join(f"- {e}" for e in errors)
    return text


def check(j: dict[str, Any], item: dict[str, Any]) -> list[str]:
    """Rubric rules a judgment can be checked against without reading the answer."""
    gold = item["gold"]
    errs: list[str] = []
    c = j.get("correctness")
    points = j.get("key_points") or []
    if c not in (0, 1, 2):
        errs.append("correctness must be 0, 1 or 2")
    if len(points) != len(gold.get("key_points") or []):
        errs.append(f"key_points must have exactly {len(gold.get('key_points') or [])} labels, one per gold key point")
    if gold.get("short_answer") is None and j.get("short_answer_match") is not None:
        errs.append("short_answer_match must be null because gold short_answer is null")
    if gold.get("short_answer") is not None and j.get("short_answer_match") is None:
        errs.append("short_answer_match must be true or false because gold short_answer is not null")
    if item.get("sources"):
        if j.get("faithfulness") not in (0, 1, 2):
            errs.append("faithfulness must be 0, 1 or 2 because sources are given")
    elif j.get("faithfulness") is not None:
        errs.append("faithfulness must be null because no sources were given")
    if not gold.get("gold_citation") and j.get("citation_matches_gold") != "not_applicable":
        errs.append("citation_matches_gold must be not_applicable because gold_citation is empty")
    if gold.get("gold_citation") and j.get("citation_matches_gold") == "not_applicable":
        errs.append("citation_matches_gold cannot be not_applicable because gold_citation is not empty")
    if bool(j.get("citations")) != bool(j.get("cites_source")):
        errs.append("cites_source must be true exactly when citations is non-empty")
    answerable, abstained = gold.get("answerable"), bool(j.get("abstained"))
    if answerable == "yes" and abstained and c != 0:
        errs.append("the answer abstains on an answerable question, so correctness must be 0")
    if answerable == "no" and not abstained and c != 0:
        errs.append("the answer does not abstain on an unanswerable question, so correctness must be 0")
    if answerable in ("yes", "partial") and points:
        if c == 2 and (j.get("contradiction") or any(p != "covered" for p in points)):
            errs.append("correctness 2 requires every key point covered and no contradiction")
        if c and c > 0 and all(p == "missing" for p in points):
            errs.append("every key point is missing, so correctness must be 0")
    return errs


def finished(j: dict[str, Any], item: dict[str, Any], model: str, errs: list[str]) -> dict[str, Any]:
    row = {key: j.get(key) for key in KEYS}
    row["id"] = item["id"]
    answerable = item["gold"].get("answerable")
    if answerable in ("yes", "no"):
        row["abstention_correct"] = bool(row["abstained"]) == (answerable == "no")
    row["judge_model"] = model
    if errs:
        row["judge_errors"] = errs
    return row


def load_sets(paths: list[Path], out_dir: Path, limit: int | None) -> list[tuple[Path, dict[str, Any]]]:
    """(output file, item) for every item not judged yet."""
    todo: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        items = json.loads(path.read_text(encoding="utf-8"))
        out_path = out_dir / f"{path.stem}.jsonl"
        done = {str(r["id"]) for r in load_records(out_path)}
        pending = [it for it in items if str(it["id"]) not in done]
        if limit is not None:
            pending = pending[:limit]
        print(f"[judge] {path.stem}: {len(items)} items, {len(done)} judged, {len(pending)} to judge", file=sys.stderr)
        todo.extend((out_path, it) for it in pending)
    return todo


def write(out_path: Path, row: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def judge_sync(client: genai.Client, todo: list[tuple[Path, dict[str, Any]]], args: argparse.Namespace) -> None:
    for i, (out_path, item) in enumerate(todo, start=1):
        errs: list[str] | None = None
        try:
            for _ in range(args.max_attempts):
                j = call_json(client, args.model, JUDGE_SYSTEM, user_prompt(item, errs), judge_schema(item), args)
                errs = check(j, item)
                if not errs:
                    break
        except Exception as e:
            print(f"[judge {i}/{len(todo)}] {out_path.stem} {item['id']} error: {str(e)[:200]}", file=sys.stderr)
            continue
        write(out_path, finished(j, item, args.model, errs or []))
        status = f"kept with {len(errs)} rule errors" if errs else "ok"
        print(f"[judge {i}/{len(todo)}] {out_path.stem} {item['id']} correctness={j.get('correctness')} {status}", file=sys.stderr)
        if args.sleep:
            time.sleep(args.sleep)


def judge_batch(client: genai.Client, todo: list[tuple[Path, dict[str, Any]]], args: argparse.Namespace) -> None:
    """Rounds of Batch API jobs (first judgments, then re-judgments with their errors).

    Attempts, errors and the pending job live in --batch-state, so a rerun resumes.
    """
    by_key = {f"{out_path.stem}::{item['id']}": (out_path, item) for out_path, item in todo}
    state: dict[str, Any] = json.loads(args.batch_state.read_text(encoding="utf-8")) if args.batch_state.is_file() else {}
    attempts: dict[str, int] = state.setdefault("attempts", {})
    errors: dict[str, list[str]] = state.setdefault("errors", {})
    done: set[str] = set()

    def save() -> None:
        args.batch_state.parent.mkdir(parents=True, exist_ok=True)
        args.batch_state.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def process(results: dict[str, dict[str, Any]]) -> None:
        for key, raw in results.items():
            if key not in by_key or key in done:
                continue
            out_path, item = by_key[key]
            attempts[key] = attempts.get(key, 0) + 1
            if "_error" in raw:
                errors[key] = []
                print(f"[batch] {key}: request failed: {raw['_error']}", file=sys.stderr)
                continue
            errs = check(raw, item)
            if errs and attempts[key] < args.max_attempts:
                errors[key] = errs
                continue
            write(out_path, finished(raw, item, args.model, errs))
            done.add(key)
            errors.pop(key, None)
            attempts.pop(key, None)

    if state.get("job"):
        print(f"[batch] resuming job {state['job']}", file=sys.stderr)
        process(wait_batch(client, state["job"], args))
        state["job"] = None
        save()

    while True:
        pending = [k for k in by_key if k not in done and attempts.get(k, 0) < args.max_attempts]
        if not pending:
            break
        reqs = {k: batch_request(JUDGE_SYSTEM, user_prompt(by_key[k][1], errors.get(k)), judge_schema(by_key[k][1]), args.temperature, args)
                for k in pending}
        groups = split_by_tokens(reqs, args.batch_max_tokens)
        for i, group in enumerate(groups, start=1):
            label = f"judge_{int(time.time())}_{i}of{len(groups)}"
            print(f"[batch] group {i}/{len(groups)}: {len(group)} requests, "
                  f"~{sum(estimated_tokens(r) for r in group.values()) / 1e6:.1f}M tokens", file=sys.stderr)
            state["job"] = submit_batch(client, args.model, group, label, args)
            save()
            process(wait_batch(client, state["job"], args))
            state["job"] = None
            save()
        print(f"[batch] judged {len(done)}/{len(by_key)}", file=sys.stderr)

    failed = [k for k in by_key if k not in done]
    if failed:
        for k in failed:
            attempts.pop(k, None)
            errors.pop(k, None)
        print(f"[batch] {len(failed)} requests failed every attempt; rerun to try them again: {', '.join(failed)}", file=sys.stderr)
    save()


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge build_eval_set.py items with a Gemini rubric.")
    ap.add_argument("sets", nargs="+", type=Path, help="judge_sets/<stem>.json files from build_eval_set.py")
    ap.add_argument("--out-dir", type=Path, default=Path("judgments"))
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--api-version", default="v1beta")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--max-attempts", type=int, default=2, help="judgments per item when the rubric checks fail")
    ap.add_argument("--max-api-retries", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=10.0, help="first wait (s) after a rate-limit/server error")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--sleep", type=float, default=0.0, help="pause (s) between items in the non-batch mode")
    ap.add_argument("--limit", type=int, default=None, help="judge at most N items per set this run")
    ap.add_argument("--batch", action="store_true", help="judge through the Batch API (half price, asynchronous)")
    ap.add_argument("--batch-dir", type=Path, default=Path("judgments/batch"), help="batch request files")
    ap.add_argument("--batch-state", type=Path, default=Path("judgments/judge_batch_state.json"))
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--batch-max-tokens", type=int, default=4_000_000,
                    help="estimated input tokens per batch job; jobs run one at a time (Tier 1 cap: 5M active per model)")
    args = ap.parse_args()

    key = os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(f"missing env {args.api_key_env}")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(api_version=args.api_version, timeout=args.timeout * 1000))
    todo = load_sets(args.sets, args.out_dir, args.limit)
    if not todo:
        return
    (judge_batch if args.batch else judge_sync)(client, todo, args)


if __name__ == "__main__":
    main()
