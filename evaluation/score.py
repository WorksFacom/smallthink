#!/usr/bin/env python3
"""Score answers against the gold set, with and without the judge.

Automatic metrics come from the answers and the gold record only: lexical
refusal check, retrieval (gold passage among the retrieved sources), binary
verdict (first word against gold short_answer), and citation (resolution +
article/considerando parsed from the answer against gold_citation). Judge
metrics summarize judge.py output. Disagreements between the two, and
judgments that break the rubric, are written as flags to review.

  python score.py                                   # every answers/*.jsonl
  python score.py --systems finetuned-rag gpt-oss-120b-rag

Writes results/scores.json (overall and per answerability, question
type, resolution and unanswerable kind) and results/flags.jsonl, and
prints a summary table.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path
from typing import Any, Callable

REFUSAL_RE = re.compile(
    r"não (é|foi) possível responder|não encontrei|não há (informaç\w+|trechos|dados) suficientes"
    r"|não (consta|constam|há) nos trechos|os trechos (normativos )?(fornecidos|disponíveis|recuperados) não",
    re.IGNORECASE,
)
VERDICT_RE = re.compile(r"[^\wÀ-ÿ]*(sim|não|nao|depende)\b", re.IGNORECASE)
RES_RE = r"(?<![\d.])(?P<num>\d\.?\d{3})\s*/\s*(?P<year>\d{4})"
CITE_RE = re.compile(
    rf"{RES_RE}|(?P<cem>c[óo]digo\s+de\s+[ée]tica\s+m[ée]dica)|(?P<law>\b(?:lei|decreto|portaria|rdc)\b)"
    r"|\bart(?:igo)?s?\.?\s*(?P<art>\d+)|\bconsiderando\s*(?P<cons>\d+)|\b(?P<ementa>ementa)\b"
    r"|\bcap[íi]tulo\s+(?P<cap>[IVXLC]+)\s*,\s*(?P<capitem>[IVXLC]+)\b|\bpre[âa]mbulo\s*,\s*(?P<pre>[IVXLC]+)\b",
    re.IGNORECASE,
)
FOLLOWING_RES = re.compile(rf"\bd[aoe]\s+(?:resolu[çc][ãa]o[^\d]{{0,20}}{RES_RE}|(?P<cem>c[óo]digo\s+de\s+[ée]tica\s+m[ée]dica))", re.IGNORECASE)
CODIGO = "2217/2018"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[Any]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def refused(answer: str) -> bool:
    return bool(REFUSAL_RE.search(answer[:400]))


def verdict(answer: str) -> str | None:
    """Sim/Não/Depende opening the answer, skipping markdown headings and table lines."""
    for line in answer.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "|", "---")):
            continue
        plain = re.sub(r"[*_`>]", "", line).strip()
        if line.startswith("**") and line.endswith("**") and len(plain.split()) <= 6 and not VERDICT_RE.match(plain):
            continue
        m = VERDICT_RE.match(plain)
        return None if not m else {"nao": "não"}.get(m.group(1).lower(), m.group(1).lower())
    return None


def gold_verdict(short_answer: str | None) -> str | None:
    m = VERDICT_RE.match(short_answer or "")
    return None if not m else {"nao": "não"}.get(m.group(1).lower(), m.group(1).lower())


def citations(text: str) -> set[tuple[str, str | None]]:
    """(resolution, "art N" | "considerando N" | "capitulo I, VII" | "preambulo V" | "ementa" | None) pairs. An article
    belongs to a resolution named right after it ("Art. 8º da Resolução nº ..."),
    otherwise to the last one named; the Código de Ética counts as 2217/2018."""
    out: set[tuple[str, str | None]] = set()
    current: str | None = None
    last_res_end = -100
    for m in CITE_RE.finditer(text or ""):
        if m.group("num"):
            current = f"{m.group('num').replace('.', '')}/{m.group('year')}"
            last_res_end = m.end()
            out.add((current, None))
        elif m.group("cem"):
            if m.start() - last_res_end > 40:
                current = CODIGO
                out.add((current, None))
        elif m.group("law"):
            current = None
        else:
            if m.group("art"):
                part = f"art {int(m.group('art'))}"
            elif m.group("cons"):
                part = f"considerando {int(m.group('cons'))}"
            elif m.group("cap"):
                part = f"capitulo {m.group('cap').upper()}, {m.group('capitem').upper()}"
            elif m.group("pre"):
                part = f"preambulo {m.group('pre').upper()}"
            else:
                part = "ementa"
            res = current
            after = FOLLOWING_RES.search(re.split(r"[;\n|]", text[m.end() : m.end() + 60], maxsplit=1)[0])
            if after:
                res = CODIGO if after.group("cem") else f"{after.group('num').replace('.', '')}/{after.group('year')}"
            if res:
                out.add((res, part))
    return out


def has_retrieval(answer: dict[str, Any] | None) -> bool:
    """Bot rows (compare.py type bot) carry top_k; closed-book rows do not."""
    return bool(answer) and "top_k" in answer


def apply_rubric(j: dict[str, Any], answerable: str) -> dict[str, Any]:
    """Judgment with the rubric's deterministic rules applied: abstention_correct
    from abstained + answerability, and correctness capped (abstaining on an
    answerable question, answering an unanswerable one, or missing every key
    point is 0; a 2 needs every key point covered and no contradiction). The
    judge's own score stays in correctness_raw."""
    j = dict(j)
    abstained = bool(j.get("abstained"))
    c = raw = j.get("correctness")
    points = j.get("key_points") or []
    if answerable == "yes":
        j["abstention_correct"] = not abstained
        if abstained:
            c = 0
    elif answerable == "no":
        j["abstention_correct"] = abstained
        if not abstained:
            c = 0
    if answerable != "no" and points and all(p == "missing" for p in points):
        c = 0
    if c == 2 and answerable != "no" and (j.get("contradiction") or any(p != "covered" for p in points)):
        c = 1
    j["correctness"], j["correctness_raw"] = c, raw
    return j


def item_row(system: str, gold: dict[str, Any], answer: dict[str, Any] | None, judgment: dict[str, Any] | None) -> dict[str, Any]:
    text = (answer or {}).get("answer") or ""
    answerable = gold["answerable"]
    if judgment:
        judgment = apply_rubric(judgment, answerable)
    row: dict[str, Any] = {
        "system": system,
        "id": gold["id"],
        "question_type": gold["question_type"],
        "answerable": answerable,
        "source_document": gold.get("source_document"),
        "unanswerable_kind": gold.get("unanswerable_kind"),
        "has_answer": bool(text.strip()) and not (answer or {}).get("error"),
        "judgment": judgment,
    }
    row["refused"] = refused(text)
    row["no_sources"] = (answer or {}).get("abstention") == "no_sources"
    gold_passages = set(gold.get("gold_passages") or [])
    if gold_passages and has_retrieval(answer):
        got = {s.get("passage_id") for s in answer.get("sources") or []}
        row["retrieval_hit"] = bool(got & gold_passages)
        row["gold_passage_recall"] = len(got & gold_passages) / len(gold_passages)
    gv = gold_verdict(gold.get("short_answer")) if gold["question_type"] == "binary" else None
    if answerable == "yes" and gv:
        mv = None if row["refused"] else verdict(text)
        row["verdict"] = mv
        row["verdict_match"] = mv == gv
    if answerable == "yes" and not row["refused"] and text:
        gold_cites = citations(gold.get("gold_citation") or "")
        got = citations(text)
        row["cites"] = bool(got)
        row["citation_resolution_match"] = bool({r for r, _ in gold_cites} & {r for r, _ in got})
        row["citation_match"] = bool({c for c in gold_cites if c[1]} & got)
    row["elapsed_seconds"] = (answer or {}).get("elapsed_seconds")
    return row


def flags(row: dict[str, Any]) -> list[str]:
    j = row["judgment"]
    out: list[str] = []
    if not row["has_answer"]:
        out.append("missing_answer")
    if j is None:
        return out + ["missing_judgment"]
    errors = [e for e in j.get("judge_errors") or [] if "abstention_correct" not in e and "correctness" not in e]
    if errors:
        out.append("judge_rule_errors: " + "; ".join(errors))
    if j["correctness"] != j["correctness_raw"]:
        out.append(f"correctness_capped ({j['correctness_raw']} -> {j['correctness']})")
    if bool(j.get("abstained")) != row["refused"]:
        out.append(f"abstention_disagreement (judge={j.get('abstained')}, lexical={row['refused']})")
    if row.get("verdict") is not None and j.get("short_answer_match") is not None and j["short_answer_match"] != row["verdict_match"]:
        out.append(f"verdict_disagreement (judge={j['short_answer_match']}, first_word={row['verdict']})")
    if "citation_match" in row:
        judge_match = j.get("citation_matches_gold") in ("full", "partial")
        if judge_match != row["citation_match"]:
            out.append(f"citation_disagreement (judge={j.get('citation_matches_gold')}, parsed={row['citation_match']})")
    return out


def share(rows: list[dict[str, Any]], key: str, pred: Callable[[Any], bool] = bool) -> float | None:
    values = [pred(r[key]) for r in rows if r.get(key) is not None]
    return mean(values)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    yes = [r for r in rows if r["answerable"] == "yes"]
    no = [r for r in rows if r["answerable"] == "no"]
    out: dict[str, Any] = {
        "n": len(rows),
        "answered": sum(r["has_answer"] for r in rows),
        "auto": {
            "refused_unanswerable": mean([r["refused"] for r in no]),
            "refused_answerable": mean([r["refused"] for r in yes]),
            "no_sources_refusals": sum(r["no_sources"] for r in rows),
            "retrieval_hit": share(yes, "retrieval_hit"),
            "gold_passage_recall": share(yes, "gold_passage_recall", lambda v: v),
            "verdict_accuracy": share(yes, "verdict_match"),
            "verdict_unclear": sum(1 for r in yes if "verdict" in r and r["verdict"] is None),
            "cites": share(yes, "cites"),
            "citation_resolution_match": share(yes, "citation_resolution_match"),
            "citation_match": share(yes, "citation_match"),
            "median_seconds": round(statistics.median(t), 1) if (t := [r["elapsed_seconds"] for r in rows if r["elapsed_seconds"] is not None]) else None,
        },
    }
    judged = [r for r in rows if r["judgment"]]
    if not judged:
        return out
    js = [r["judgment"] for r in judged]
    js_yes = [r["judgment"] for r in judged if r["answerable"] == "yes"]
    js_no = [r["judgment"] for r in judged if r["answerable"] == "no"]
    points = [p for j in js_yes for p in j.get("key_points") or []]
    statuses = collections.Counter(c.get("status") for j in js for c in j.get("citations") or [])
    out["judge"] = {
        "judged": len(judged),
        "correctness_mean": mean([j.get("correctness") for j in js]),
        "correctness_raw_mean": mean([j.get("correctness_raw") for j in js]),
        "correctness_capped": sum(1 for j in js if j.get("correctness") != j.get("correctness_raw")),
        "correctness_2": mean([j.get("correctness") == 2 for j in js]),
        "correctness_0": mean([j.get("correctness") == 0 for j in js]),
        "key_point_coverage": mean([{"covered": 1, "partial": 0.5}.get(p, 0) for p in points]),
        "short_answer_match": mean([j.get("short_answer_match") for j in js_yes]),
        "abstention_correct": mean([j.get("abstention_correct") for j in js]),
        "abstained_unanswerable": mean([j.get("abstained") for j in js_no]),
        "abstained_answerable": mean([j.get("abstained") for j in js_yes]),
        "citation_matches_gold": dict(collections.Counter(j.get("citation_matches_gold") for j in js_yes)),
        "citation_status": dict(statuses),
        "faithfulness_mean": mean([j.get("faithfulness") for j in js if not j.get("abstained")]),
        "contradiction": mean([j.get("contradiction") for j in js]),
        "must_not_violated": mean([bool(j.get("must_not_violated")) for j in js]),
        "language_pt": mean([j.get("language_pt") for j in js]),
        "foreign_law": mean([j.get("foreign_law") for j in js]),
        "with_other_rule_errors": sum(
            1 for j in js if any("abstention_correct" not in e and "correctness" not in e for e in j.get("judge_errors") or [])
        ),
    }
    return out


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            groups[str(r[key])].append(r)
    return {k: metrics(v) for k, v in sorted(groups.items())}


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def summary_table(scores: dict[str, Any]) -> str:
    cols = [
        ("judged", lambda m: str(m.get("judge", {}).get("judged", 0))),
        ("correctness (0-2)", lambda m: "-" if not m.get("judge") else f"{m['judge']['correctness_mean']:.2f}"),
        ("fully correct", lambda m: pct(m.get("judge", {}).get("correctness_2"))),
        ("key points", lambda m: pct(m.get("judge", {}).get("key_point_coverage"))),
        ("verdict (judge)", lambda m: pct(m.get("judge", {}).get("short_answer_match"))),
        ("verdict (first word)", lambda m: pct(m["auto"]["verdict_accuracy"])),
        ("refuses unanswerable", lambda m: pct(m.get("judge", {}).get("abstained_unanswerable", m["auto"]["refused_unanswerable"]))),
        ("refuses answerable", lambda m: pct(m.get("judge", {}).get("abstained_answerable", m["auto"]["refused_answerable"]))),
        ("citation (parsed)", lambda m: pct(m["auto"]["citation_match"])),
        ("faithfulness (0-2)", lambda m: "-" if m.get("judge", {}).get("faithfulness_mean") is None else f"{m['judge']['faithfulness_mean']:.2f}"),
        ("retrieval hit", lambda m: pct(m["auto"]["retrieval_hit"])),
    ]
    lines = ["| system | n | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for system, s in scores.items():
        m = s["overall"]
        lines.append(f"| {system} | {m['n']} | " + " | ".join(f(m) for _, f in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Automatic + judge scores per system.")
    ap.add_argument("--gold", type=Path, default=Path("questions/gold.jsonl"))
    ap.add_argument("--answers-dir", type=Path, default=Path("answers"))
    ap.add_argument("--judgments-dir", type=Path, default=Path("judgments"))
    ap.add_argument("--systems", nargs="*", help="answer file stems (default: every *.jsonl in --answers-dir)")
    ap.add_argument("--scores", type=Path, default=Path("results/scores.json"))
    ap.add_argument("--flags", type=Path, default=Path("results/flags.jsonl"))
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    systems = args.systems or sorted(p.stem for p in args.answers_dir.glob("*.jsonl"))
    scores: dict[str, Any] = {}
    flagged: list[dict[str, Any]] = []
    for system in systems:
        answers = {str(r["id"]): r for r in load_jsonl(args.answers_dir / f"{system}.jsonl")}
        judgments = {str(r["id"]): r for r in load_jsonl(args.judgments_dir / f"{system}.jsonl")}
        rows = [item_row(system, g, answers.get(g["id"]), judgments.get(g["id"])) for g in gold]
        rows = [r for r in rows if r["has_answer"] or r["judgment"]] if answers else rows
        scores[system] = {
            "overall": metrics(rows),
            "by_answerable": grouped(rows, "answerable"),
            "by_question_type": grouped([r for r in rows if r["answerable"] == "yes"], "question_type"),
            "by_source_document": grouped([r for r in rows if r["answerable"] == "yes"], "source_document"),
            "by_unanswerable_kind": grouped(rows, "unanswerable_kind"),
        }
        for r in rows:
            for f in flags(r):
                if f != "missing_judgment" or judgments:
                    flagged.append({"system": system, "id": r["id"], "flag": f})

    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.flags.open("w", encoding="utf-8") as f:
        for row in flagged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(summary_table(scores))
    counts = collections.Counter((r["system"], r["flag"].split(" ")[0].rstrip(":")) for r in flagged)
    if counts:
        print("\nflags (review in", args.flags, "):")
        for (system, kind), n in sorted(counts.items()):
            print(f"  {system}: {kind} {n}")
    print(f"\nscores -> {args.scores}")


if __name__ == "__main__":
    main()
