#!/usr/bin/env python3
"""Fresh evaluation set: new questions + gold records, drafted with Gemini.

Three stages (default: all, each resumable):

  generate  new questions from the current passages. Answerable questions are
            written per resolution (quota proportional to its passages, with a
            minimum each) from that resolution's own passages; unanswerable
            ones from the resolution topic list only, about topics outside the
            knowledge base. Over-generates, then drops questions that name the
            norm, repeat each other, or resemble a training Q&A / an older
            test question (word-set Jaccard).

  draft     the gold record of every candidate, independently of how it was
            written: route the question to up to 3 resolutions, then draft from
            ALL their passages, with passage_id restricted to those passages by
            the response schema. Quotes must be exact substrings, key points
            must be supported; failing drafts are retried with the errors.
            Routing runs on --route-model (Flash by default). Prompts put the
            shared text first (topic list; passages in a fixed resolution
            order) and the question last, so repeated prefixes can be billed
            at the cached-input rate. --batch sends routing and each draft
            round as one Batch API job (half price, results in minutes to
            hours); progress is kept in --batch-state, so a rerun resumes.

  finalize  pick --total questions: answerable candidates the drafter judged
            yes/partial (per-resolution quotas, gold matching the passages the
            question was written from first) and --unanswerable-ratio of
            candidates the drafter confirmed as "no". Writes the gold set, the
            plain question list for compare.py, and a report with the records
            to review by hand first.

  export GEMINI_API_KEY=...
  python build_gold.py --stage generate --total 20      # small trial
  python build_gold.py                                   # 200 questions, 10% unanswerable
  python build_gold.py --batch                           # same, drafting via the Batch API
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

EVIDENCE_KINDS = {"ementa", "considerando", "artigo", "item"}
# Articles suspended by a court decision (e.g. Resolução 2232/2019, Art. 5, 6, 10)
# are not rules in force: no question or gold record may rely on them.
RE_SUSPENDED = re.compile(r"artigo\s+suspenso\s+por\s+decis[ãa]o\s+judicial", re.IGNORECASE)
GOLD_FIELDS = (
    "id", "question", "question_type", "answerable", "short_answer", "reference",
    "key_points", "gold_passages", "gold_citation", "evidence", "must_not",
    "confidence", "notes",
)
QUESTION_TYPES = ["binary", "factual", "list", "explanatory"]
UNANSWERABLE_KINDS = ("near_miss", "out_of_scope")
NO_ANSWER_REFERENCE = "Não é possível responder com base nos trechos normativos disponíveis."
RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504", "UNAVAILABLE", "DEADLINE", "timed out")
# Network failures (httpx/httpcore ReadTimeout, ConnectError, RemoteProtocolError...)
RETRYABLE_EXCEPTIONS = ("Timeout", "Connect", "RemoteProtocol", "Network", "Protocol")
RE_CITED = re.compile(
    r"Resolu[çc][ãa]o\s+(?:CFM|n[º°o])|\bArt(?:igo)?\.?\s*\d+|\bconsiderando\s*\d+|\b\d{3,4}/\d{4}\b", re.IGNORECASE
)
STOPWORDS = {
    "a", "o", "as", "os", "e", "é", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "um", "uma", "para", "por", "pelo", "pela", "com", "sem", "que", "se", "ou", "ao", "aos", "à",
    "seu", "sua", "seus", "suas", "ser", "são", "qual", "quais", "quem", "quando", "como", "mais",
    "não", "pode", "deve", "há", "caso", "este", "esta", "esse", "essa", "isso", "sobre", "entre",
}

# Routing/generation descriptions; resolutions not listed fall back to their ementa.
RESOLUTION_TOPICS = {
    "cfm_res_1805_2006": "Limitação ou suspensão de procedimentos e tratamentos que prolonguem a vida na fase terminal de doenças graves e incuráveis; cuidados paliativos.",
    "cfm_res_1821_2007": "Prontuário do paciente: digitalização, prontuário eletrônico, certificação de sistemas, guarda, prazo de preservação e propriedade.",
    "cfm_res_1995_2012": "Diretivas antecipadas de vontade dos pacientes.",
    "cfm_res_1999_2012": "Terapias hormonais e medicina antienvelhecimento; procedimentos sem comprovação científica.",
    "cfm_res_2004_2012": "Prática ortomolecular: vitaminas, minerais, antioxidantes, quelação e procedimentos sem evidência.",
    "cfm_res_2217_2018": "Código de Ética Médica: princípios fundamentais, direitos dos médicos, responsabilidade profissional, direitos humanos, relação com pacientes e familiares, relação entre médicos, remuneração, sigilo profissional, documentos médicos (atestados, prontuário, receitas, laudos), auditoria e perícia, ensino e pesquisa, publicidade.",
    "cfm_res_2232_2019": "Recusa terapêutica por pacientes e objeção de consciência na relação médico-paciente.",
    "cfm_res_2314_2022": "Telemedicina: teleconsulta, telediagnóstico, telemonitoramento, prescrição a distância, registro e guarda de dados.",
    "cfm_res_2327_2022": "Aplicação de terapêuticas reconhecidas; novos procedimentos e uso experimental.",
    "cfm_res_2333_2023": "Prescrição de esteroides androgênicos e anabolizantes para fins estéticos ou ganho de desempenho.",
    "cfm_res_2336_2023": "Publicidade e propaganda médicas: redes sociais, uso de imagem de pacientes, divulgação de especialidade (RQE), proibições e permissões.",
}

GEN_SYSTEM = """You write evaluation questions for a question-answering assistant on Brazilian medical ethics and medical law (normas do Conselho Federal de Medicina, CFM). The questions will be used to test the assistant, so each one must be answerable ONLY from the PASSAGES given and must read like a real question from a physician, patient, medical student or clinic manager.

Rules:
1. Write in Brazilian Portuguese, one question per item, at most 60 words (a short practical scenario is allowed).
2. Each question must be fully answerable from the given passages; list in "target_passages" the 1–3 passage_ids needed.
3. Never mention resolution numbers, article numbers, "considerando", or phrases like "segundo o texto" or "conforme a resolução". Name the topic in plain words.
4. Mix question types: about 45% "binary" (pode / é permitido / é ético / deve), 15% "factual" (one specific fact: prazo, órgão, documento), 15% "list" (several items or situations), 25% "explanatory" (explain a rule, a difference or a procedure).
5. Vary the wording: no more than a quarter of the questions may start with the same two words; use practical cases ("Um paciente...", "Durante uma teleconsulta...", "Uma clínica quer...").
6. Prefer substantive rules, conditions and exceptions, and include some questions that need two passages. Avoid trivia about dates, publication, entry into force or revocation.
7. Balance the correct answers of "binary" questions: about one third "Sim" (something the passages allow, require or guarantee), about one fifth "Depende" (the passages set conditions or exceptions), the rest "Não". Many passages are prohibitions ("É vedado ao médico..."): to get a "Sim", ask about what they protect or require (e.g. "O paciente tem direito a...?", "O médico deve...?", "É permitido... quando...?").
8. Do not repeat or paraphrase any question listed under ALREADY WRITTEN.
Return ONLY the JSON object defined by the response schema."""

UNANS_SYSTEM = """You write evaluation questions that a Brazilian medical-ethics assistant must REFUSE because its knowledge base does not contain the answer. The knowledge base contains ONLY the CFM resolutions listed under KNOWLEDGE BASE. Write realistic questions from physicians, patients or clinic managers, in two kinds (set "unanswerable_kind"):

- "near_miss" (about half): on a topic the knowledge base DOES cover (telemedicina, prontuário, publicidade médica, recusa terapêutica, diretivas antecipadas, terapias hormonais, sigilo, atestados...), but asking for a specific detail those resolutions do not set: a number, deadline, percentage, fee, form, technical standard, procedure step, or a case they do not regulate. The topic must look covered, so only checking the actual rules reveals that the answer is missing.
- "out_of_scope" (the rest): medical ethics, medical law or medical practice in Brazil outside these resolutions (other CFM resolutions, federal laws, health-plan (ANS) rules, clinical protocols, labor or tax rules for physicians).

Rules:
1. Brazilian Portuguese, at most 60 words each.
2. Same tone as normal user questions. Never ask something the listed resolutions answer, including the broad Código de Ética Médica (sigilo, documentos médicos, publicidade, remuneração, pesquisa, relação com pacientes, responsabilidade profissional): if unsure whether a detail is regulated, choose a more specific detail.
3. No resolution or article numbers.
4. Mix question types: binary, factual, list and explanatory.
5. Do not repeat or paraphrase any question listed under ALREADY WRITTEN.
Return ONLY the JSON object defined by the response schema."""

ROUTE_SYSTEM = (
    "You route questions about Brazilian medical ethics to the CFM resolutions that may "
    "contain the answer. Return the document_ids of every resolution that could contain the "
    "rule, condition or fact asked for, most relevant first, at most {max_docs}. Include "
    "cfm_res_2217_2018 (Código de Ética Médica) whenever the question concerns general "
    "professional conduct, medical documents, confidentiality, the patient relationship, "
    "fees, publicity or research. Return an empty list only if no resolution is plausible."
)

DRAFT_SYSTEM = """You are building a GOLD-STANDARD evaluation record for a question-answering system about Brazilian medical ethics and medical law (normas do Conselho Federal de Medicina, CFM).

You receive ONE question and CANDIDATE PASSAGES: all passages of the resolutions that may answer it. The candidate passages are the ONLY allowed knowledge source.

## Hard rules
1. Use ONLY the candidate passages. Do not use your own knowledge of CFM resolutions or Brazilian law.
2. Every "passage_id" must be one of the candidate passage_ids, copied exactly.
3. Every "quote" must be an exact substring of that passage's "text" (copy-paste; you may only collapse line breaks into single spaces). Do not paraphrase, fix typos or add words. Keep quotes short: the sentence or clause that supports the point.
4. "gold_citation" must be built only from the "citation" field of the passages you used, joined with "; ".
5. Every key point must be supported by at least one evidence quote ("supports" lists the 0-based indexes of the key points it supports).
6. Prefer "artigo" and "item" passages over "considerando" passages when both support the same point.
7. Answerability:
   - "yes": the candidate passages fully answer the question.
   - "partial": they answer only part of it; say in "notes" what is missing.
   - "no": none of them answers it; leave gold_passages, evidence and key_points empty, set short_answer to null, use the reference "Não é possível responder com base nos trechos normativos disponíveis." and explain in "notes" which information is missing.
8. Do not mark a question "yes" only because the topic is similar: the passage must contain the actual rule, condition or fact asked for.
9. Write "reference", "key_points", "must_not" and "notes" in Brazilian Portuguese.

## Fields
- question_type: "binary" (pode / é permitido / é ético / deve), "factual" (one specific fact), "list" (several items), "explanatory" (explain, compare, describe a procedure).
- short_answer: binary → "Sim", "Não" or "Depende"; factual → the core fact only; list and explanatory → null; answerable "no" → null.
- reference: 1–3 objective sentences written only from the evidence.
- key_points: 1–5 atomic statements a correct answer must contain, including required conditions or exceptions.
- gold_passages: the passage_ids used in evidence.
- must_not: claims or citations a correct answer must not make (e.g. "citar HIPAA ou legislação estrangeira", or the most likely wrong conclusion).
- confidence: "high", "medium" or "low" (low when the passages are indirect or need interpretation).

Return ONLY the JSON object defined by the response schema."""


# ---------------------------------------------------------------- data helpers


def load_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".json":
        data = json.loads(text)
        return data if isinstance(data, list) else data.get("questions", [])
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[^\W\d_]+", (text or "").lower()) if len(w) > 2 and w not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def load_corpus(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Evidence-eligible passages per document (no duplicates, no suspended
    articles), and a topic line per document."""
    by_doc: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    ementas: dict[str, str] = {}
    suspended: list[str] = []
    for p in load_records(path):
        if p.get("duplicate_of") or p.get("kind") not in EVIDENCE_KINDS:
            continue
        if RE_SUSPENDED.search(p.get("text") or ""):
            suspended.append(p["citation"])
            continue
        by_doc[p["document_id"]].append(p)
        if p["kind"] == "ementa":
            ementas[p["document_id"]] = norm(p["text"])
    topics = {}
    for doc, passages in by_doc.items():
        base = passages[0]["citation"].split(",")[0].split(" —")[0]
        topic = RESOLUTION_TOPICS.get(doc) or ementas.get(doc) or norm(passages[0]["text"])[:300]
        topics[doc] = f"{base}: {topic}"
    if suspended:
        print(f"excluded {len(suspended)} suspended articles: {'; '.join(suspended)}", file=sys.stderr)
    return dict(by_doc), topics


def packet(passages: list[dict[str, Any]]) -> str:
    return json.dumps(
        [{"passage_id": p["passage_id"], "kind": p["kind"], "citation": p["citation"], "text": p["text"]} for p in passages],
        ensure_ascii=False,
    )


def allocate(total: int, sizes: dict[str, int], minimum: int) -> dict[str, int]:
    """Largest-remainder split of total proportional to sizes, at least `minimum` each."""
    keys = [k for k in sizes if sizes[k] > 0]
    if not keys or total <= 0:
        return {k: 0 for k in sizes}
    minimum = min(minimum, total // len(keys))
    out = {k: minimum for k in keys}
    rest = total - minimum * len(keys)
    weight = sum(sizes[k] for k in keys)
    shares = {k: rest * sizes[k] / weight for k in keys}
    for k in keys:
        out[k] += int(shares[k])
    left = total - sum(out.values())
    for k in sorted(keys, key=lambda k: shares[k] - int(shares[k]), reverse=True)[:left]:
        out[k] += 1
    return out


def split_contiguous(items: list[Any], n: int) -> list[list[Any]]:
    n = max(1, min(n, len(items)))
    size = math.ceil(len(items) / n)
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------- Gemini


def call_json(
    client: genai.Client,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    args: argparse.Namespace,
    temperature: float | None = None,
) -> dict[str, Any]:
    """One structured-output call, retried with backoff on rate limits, server
    errors and network failures (timeouts, dropped connections)."""
    delay = args.backoff
    for attempt in range(args.max_api_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=args.temperature if temperature is None else temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                    max_output_tokens=args.max_output_tokens,
                ),
            )
            return json.loads(resp.text or "")
        except Exception as e:
            msg = str(e)
            retryable = any(code in msg for code in RETRYABLE) or any(
                name in type(err).__name__ for err in (e, e.__cause__) if err for name in RETRYABLE_EXCEPTIONS
            )
            if attempt < args.max_api_retries and retryable:
                print(f"  api error ({type(e).__name__}: {msg[:80]}), retry in {delay:.0f}s", file=sys.stderr)
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def questions_schema(passage_ids: list[str] | None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "question": {"type": "STRING"},
        "question_type": {"type": "STRING", "enum": QUESTION_TYPES},
    }
    required = ["question", "question_type"]
    if passage_ids is not None:
        item["target_passages"] = {"type": "ARRAY", "items": {"type": "STRING", "enum": passage_ids}}
        required.append("target_passages")
    else:
        item["unanswerable_kind"] = {"type": "STRING", "enum": list(UNANSWERABLE_KINDS)}
        required.append("unanswerable_kind")
    return {
        "type": "OBJECT",
        "properties": {"questions": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": item, "required": required}}},
        "required": ["questions"],
    }


def route_schema(doc_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "document_ids": {"type": "ARRAY", "items": {"type": "STRING", "enum": doc_ids}},
            "reason": {"type": "STRING"},
        },
        "required": ["document_ids", "reason"],
    }


def draft_schema(passage_ids: list[str]) -> dict[str, Any]:
    pid = {"type": "STRING", "enum": passage_ids}
    return {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "STRING"},
            "question": {"type": "STRING"},
            "question_type": {"type": "STRING", "enum": QUESTION_TYPES},
            "answerable": {"type": "STRING", "enum": ["yes", "partial", "no"]},
            "short_answer": {"type": "STRING", "nullable": True},
            "reference": {"type": "STRING"},
            "key_points": {"type": "ARRAY", "items": {"type": "STRING"}},
            "gold_passages": {"type": "ARRAY", "items": pid},
            "gold_citation": {"type": "STRING"},
            "evidence": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "passage_id": pid,
                        "quote": {"type": "STRING"},
                        "supports": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    },
                    "required": ["passage_id", "quote", "supports"],
                },
            },
            "must_not": {"type": "ARRAY", "items": {"type": "STRING"}},
            "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]},
            "notes": {"type": "STRING"},
        },
        "required": list(GOLD_FIELDS),
    }


# ---------------------------------------------------------------- generate


def generate(client: genai.Client, corpus: dict[str, list[dict[str, Any]]], topics: dict[str, str], args: argparse.Namespace) -> None:
    if args.candidates.is_file() and not args.force:
        print(f"{args.candidates} exists; skipping generate (use --force to regenerate)", file=sys.stderr)
        return
    if args.force and args.generation_log.is_file():
        args.generation_log.unlink()
    n_ans = round(args.total * (1 - args.unanswerable_ratio))
    n_unans = args.total - n_ans
    quotas = allocate(math.ceil(n_ans * args.overgenerate), {d: len(ps) for d, ps in corpus.items()}, args.min_per_document)

    # Every finished call is appended to --generation-log under a key that only
    # depends on corpus and arguments, so a rerun reuses finished chunks.
    logged = {r["key"]: r["questions"] for r in load_records(args.generation_log)}
    if logged:
        print(f"[generate] reusing {len(logged)} chunks from {args.generation_log}", file=sys.stderr)
    args.generation_log.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.generation_log.open("a", encoding="utf-8")

    def run_chunk(key: str, system: str, user: str, schema: dict[str, Any]) -> list[dict[str, Any]]:
        if key in logged:
            return logged[key]
        questions = call_json(client, args.model, system, user, schema, args, temperature=args.gen_temperature).get("questions") or []
        checkpoint.write(json.dumps({"key": key, "questions": questions}, ensure_ascii=False) + "\n")
        checkpoint.flush()
        return questions

    raw: list[dict[str, Any]] = []
    try:
        for doc in sorted(corpus):
            if not quotas.get(doc):
                continue
            chunks = split_contiguous(corpus[doc], math.ceil(quotas[doc] / args.per_call))
            chunk_quota = allocate(quotas[doc], {str(i): len(c) for i, c in enumerate(chunks)}, 0)
            written: list[str] = []
            for i, chunk in enumerate(chunks):
                n = chunk_quota[str(i)]
                if not n:
                    continue
                user = (
                    f"Write {n} questions.\n\nRESOLUTION: {topics[doc]}\n\nPASSAGES:\n{packet(chunk)}\n\n"
                    f"ALREADY WRITTEN:\n" + ("\n".join(f"- {q}" for q in written) or "(none)")
                )
                key = f"{doc}#part{i + 1}of{len(chunks)}#n{n}"
                questions = run_chunk(key, GEN_SYSTEM, user, questions_schema([p["passage_id"] for p in chunk]))
                for q in questions:
                    written.append(q.get("question", ""))
                    raw.append({**q, "intended": "answerable", "source_document": doc})
                print(f"[generate] {doc} part {i + 1}/{len(chunks)}: {len(questions)} questions", file=sys.stderr)

        kb = "\n".join(f"- {topics[d]}" for d in sorted(corpus))
        target_unans = math.ceil(n_unans * args.unanswerable_overgenerate)
        written_unans: list[str] = []
        batch_no = 0
        while len(written_unans) < target_unans:
            batch_no += 1
            n = min(args.per_call, target_unans - len(written_unans))
            user = (
                f"Write {n} questions.\n\nKNOWLEDGE BASE:\n{kb}\n\nALREADY WRITTEN:\n"
                + ("\n".join(f"- {q}" for q in written_unans) or "(none)")
            )
            batch = run_chunk(f"unanswerable#batch{batch_no}#n{n}", UNANS_SYSTEM, user, questions_schema(None))
            if not batch:
                break
            for q in batch:
                written_unans.append(q.get("question", ""))
                raw.append({**q, "intended": "unanswerable", "source_document": None, "target_passages": []})
            print(f"[generate] unanswerable: {len(written_unans)}/{target_unans}", file=sys.stderr)
    finally:
        checkpoint.close()

    existing = [content_words(r.get("question", "")) for r in load_records(args.training_qa)]
    existing += [content_words(r.get("question", "")) for r in load_records(args.previous_questions)]
    kept: list[dict[str, Any]] = []
    kept_words: list[set[str]] = []
    dropped: collections.Counter[str] = collections.Counter()
    for q in raw:
        text = norm(q.get("question", ""))
        if not text:
            dropped["empty"] += 1
            continue
        if RE_CITED.search(text):
            dropped["names_the_norm"] += 1
            continue
        words = content_words(text)
        if any(jaccard(words, w) >= args.similarity_threshold for w in kept_words):
            dropped["near_duplicate_new"] += 1
            continue
        if any(jaccard(words, w) >= args.similarity_threshold for w in existing):
            dropped["similar_to_training_or_old_test"] += 1
            continue
        kept_words.append(words)
        kept.append({
            "id": f"c{len(kept) + 1:04d}",
            "question": text,
            "question_type": q.get("question_type"),
            "intended": q["intended"],
            "unanswerable_kind": q.get("unanswerable_kind"),
            "source_document": q.get("source_document"),
            "target_passages": q.get("target_passages") or [],
        })
    write_jsonl(args.candidates, kept)
    print(json.dumps({"generated": len(raw), "kept": len(kept), "dropped": dict(dropped),
                      "kept_by_intent": dict(collections.Counter(k["intended"] for k in kept)),
                      "kept_unanswerable_kind": dict(collections.Counter(k["unanswerable_kind"] for k in kept if k["intended"] == "unanswerable"))},
                     ensure_ascii=False))


# ---------------------------------------------------------------- draft


def validate(rec: dict[str, Any], question: dict[str, Any], texts: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    """Normalized record and the list of problems that require a retry."""
    errors: list[str] = []
    out = {field: rec.get(field) for field in GOLD_FIELDS}
    out["id"], out["question"] = question["id"], question["question"]
    for field in ("key_points", "gold_passages", "evidence", "must_not"):
        out[field] = list(out.get(field) or [])
    for field in ("reference", "gold_citation", "notes"):
        out[field] = (out.get(field) or "").strip()

    answerable, qtype = out.get("answerable"), out.get("question_type")
    if answerable == "no":
        if out["evidence"] or out["key_points"]:
            errors.append('answerable is "no" but evidence/key_points are not empty')
        out.update(short_answer=None, gold_passages=[], gold_citation="", reference=NO_ANSWER_REFERENCE)
        return out, errors

    if not out["evidence"]:
        errors.append(f'answerable is "{answerable}" but evidence is empty')
    if not out["key_points"]:
        errors.append(f'answerable is "{answerable}" but key_points is empty')
    supported: set[int] = set()
    for i, ev in enumerate(out["evidence"]):
        pid, quote = ev.get("passage_id"), norm(ev.get("quote"))
        if pid not in texts:
            errors.append(f"evidence[{i}].passage_id {pid!r} is not a candidate passage")
            continue
        if not quote or quote not in texts[pid]:
            errors.append(f"evidence[{i}].quote is not an exact substring of {pid}: {quote[:100]!r}")
        for s in ev.get("supports") or []:
            if not isinstance(s, int) or not 0 <= s < len(out["key_points"]):
                errors.append(f"evidence[{i}].supports has invalid index {s!r}")
            else:
                supported.add(s)
    missing = [i for i in range(len(out["key_points"])) if i not in supported]
    if missing:
        errors.append(f"key_points {missing} are not supported by any evidence")
    if not out["gold_citation"]:
        errors.append("gold_citation is empty")

    if qtype == "binary" and out.get("short_answer") not in ("Sim", "Não", "Depende"):
        errors.append('binary question needs short_answer "Sim", "Não" or "Depende"')
    if qtype in ("list", "explanatory"):
        out["short_answer"] = None
    if qtype == "factual" and not (out.get("short_answer") or "").strip():
        errors.append("factual question needs a short_answer")

    ordered: list[str] = []
    for ev in out["evidence"]:
        if ev.get("passage_id") in texts and ev["passage_id"] not in ordered:
            ordered.append(ev["passage_id"])
    out["gold_passages"] = ordered
    return out, errors


def doc_order(corpus: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Fixed packet order, largest resolution first: prompts that share the
    leading resolution share a long identical prefix that Gemini can cache."""
    return sorted(corpus, key=lambda d: (-len(corpus[d]), d))


def route_prompt(question: dict[str, Any], corpus: dict[str, list[dict[str, Any]]], topics: dict[str, str], args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    doc_ids = sorted(corpus)
    user = "RESOLUTIONS:\n" + "\n".join(f"- {d}: {topics[d]}" for d in doc_ids) + f"\n\nQUESTION:\n{question['question']}"
    return ROUTE_SYSTEM.format(max_docs=args.max_documents), user, route_schema(doc_ids)


def selected_documents(route: dict[str, Any], corpus: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> list[str]:
    chosen = [d for d in dict.fromkeys(route.get("document_ids") or []) if d in corpus][: args.max_documents]
    if not chosen:
        chosen = ["cfm_res_2217_2018"] if "cfm_res_2217_2018" in corpus else sorted(corpus)[:1]
    order = doc_order(corpus)
    return sorted(chosen, key=order.index)


def draft_prompt(
    question: dict[str, Any], selected: list[str], corpus: dict[str, list[dict[str, Any]]], errors: list[str] | None
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Passages first and the question last, so the shared part is a prefix."""
    candidates = [p for doc in selected for p in corpus[doc]]
    texts = {p["passage_id"]: norm(p["text"]) for p in candidates}
    user = (
        f"CANDIDATE PASSAGES:\n{packet(candidates)}\n\n"
        f"(all passages of {', '.join(selected)})\n\n"
        f"QUESTION:\n{json.dumps({'id': question['id'], 'question': question['question']}, ensure_ascii=False)}"
    )
    if errors:
        user += "\n\nYour previous answer was rejected for these reasons:\n- " + "\n- ".join(errors) + "\nReturn a corrected record."
    return user, draft_schema(list(texts)), texts


def write_draft(out: Any, question: dict[str, Any], record: dict[str, Any], selected: list[str]) -> None:
    meta = {k: question.get(k) for k in ("intended", "unanswerable_kind", "source_document", "target_passages")}
    out.write(json.dumps({**record, **meta, "selected_documents": selected}, ensure_ascii=False) + "\n")
    out.flush()


def draft_one(client: genai.Client, question: dict[str, Any], corpus: dict[str, list[dict[str, Any]]], topics: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    system, user, schema = route_prompt(question, corpus, topics, args)
    route = call_json(client, args.route_model or args.model, system, user, schema, args)
    selected = selected_documents(route, corpus, args)
    log: dict[str, Any] = {"id": question["id"], "selected_documents": selected, "route_reason": route.get("reason"), "attempts": []}
    errors: list[str] | None = None
    record: dict[str, Any] = {}
    for attempt in range(1, args.max_attempts + 1):
        user, schema, texts = draft_prompt(question, selected, corpus, errors)
        record, errors = validate(call_json(client, args.model, DRAFT_SYSTEM, user, schema, args), question, texts)
        log["attempts"].append({"attempt": attempt, "errors": errors})
        if not errors:
            log.update(status="accepted", record=record)
            return log
    log.update(status="failed", record=record)
    return log


# ---------------------------------------------------------------- draft (Batch API)

BATCH_DONE = {"JOB_STATE_SUCCEEDED"}
BATCH_FAILED = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def batch_request(system: str, user: str, schema: dict[str, Any], temperature: float, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "system_instruction": {"parts": [{"text": system}]},
        "generation_config": {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": schema,
            "max_output_tokens": args.max_output_tokens,
        },
    }


def submit_batch(client: genai.Client, model: str, requests: dict[str, dict[str, Any]], label: str, args: argparse.Namespace) -> str:
    args.batch_dir.mkdir(parents=True, exist_ok=True)
    path = args.batch_dir / f"{label}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for key, request in requests.items():
            f.write(json.dumps({"key": key, "request": request}, ensure_ascii=False) + "\n")
    uploaded = client.files.upload(file=str(path), config=types.UploadFileConfig(display_name=label, mime_type="jsonl"))
    delay = max(args.backoff, args.poll_seconds)
    for attempt in range(args.max_api_retries + 1):
        try:
            job = client.batches.create(model=model, src=uploaded.name, config={"display_name": label})
            break
        except Exception as e:
            msg = str(e)
            if attempt < args.max_api_retries and any(code in msg for code in RETRYABLE):
                print(f"[batch] create refused ({msg[:100]}), retry in {delay:.0f}s", file=sys.stderr)
                time.sleep(delay)
                delay *= 2
                continue
            raise
    print(f"[batch] submitted {label}: {len(requests)} requests -> {job.name}", file=sys.stderr)
    return job.name


def estimated_tokens(request: dict[str, Any]) -> int:
    """Rough input tokens of a request (JSON-heavy Portuguese: ~3 characters per token)."""
    return len(json.dumps(request, ensure_ascii=False)) // 3


def split_by_tokens(requests: dict[str, dict[str, Any]], budget: int) -> list[dict[str, dict[str, Any]]]:
    """Consecutive groups whose estimated tokens stay under budget (Tier 1 caps
    the tokens of active batch jobs per model, e.g. 5M for gemini-3.1-pro)."""
    groups: list[dict[str, dict[str, Any]]] = []
    current: dict[str, dict[str, Any]] = {}
    used = 0
    for key, request in requests.items():
        size = estimated_tokens(request)
        if current and used + size > budget:
            groups.append(current)
            current, used = {}, 0
        current[key] = request
        used += size
    if current:
        groups.append(current)
    return groups


def response_text(response: dict[str, Any]) -> str:
    parts = (((response.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def wait_batch(client: genai.Client, name: str, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Poll a batch job; returns {key: parsed JSON or {"_error": message}}."""
    while True:
        job = client.batches.get(name=name)
        state = getattr(job.state, "name", str(job.state))
        if state in BATCH_DONE:
            break
        if state in BATCH_FAILED:
            raise RuntimeError(f"batch {name} ended in {state}: {getattr(job, 'error', '')}")
        print(f"[batch] {name} {state}; next check in {args.poll_seconds:.0f}s", file=sys.stderr)
        time.sleep(args.poll_seconds)
    content = client.files.download(file=job.dest.file_name)
    results: dict[str, dict[str, Any]] = {}
    for line in content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        key = obj.get("key")
        if obj.get("error") or "response" not in obj:
            results[key] = {"_error": str(obj.get("error") or "no response")[:300]}
            continue
        try:
            results[key] = json.loads(response_text(obj["response"]))
        except json.JSONDecodeError as e:
            results[key] = {"_error": f"invalid JSON: {e}"}
    return results


def draft_batch(corpus: dict[str, list[dict[str, Any]]], topics: dict[str, str], args: argparse.Namespace, client: genai.Client,
                todo: list[dict[str, Any]], done: set[str]) -> None:
    """Route all, then draft in rounds (one batch job each); failures retry with their errors.

    Progress lives in --batch-state, so an interrupted run resumes the pending job.
    """
    by_id = {q["id"]: q for q in todo}
    state: dict[str, Any] = json.loads(args.batch_state.read_text(encoding="utf-8")) if args.batch_state.is_file() else {}
    routes: dict[str, dict[str, Any]] = state.setdefault("routes", {})
    errors: dict[str, list[str]] = state.setdefault("errors", {})
    attempts: dict[str, int] = state.setdefault("attempts", {})

    def save() -> None:
        args.batch_state.parent.mkdir(parents=True, exist_ok=True)
        args.batch_state.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    out = args.drafts.open("a", encoding="utf-8")
    log = args.log.open("a", encoding="utf-8")

    def process(kind: str, results: dict[str, dict[str, Any]]) -> None:
        for qid, raw in results.items():
            q = by_id.get(qid)
            if q is None or qid in done:
                continue
            if kind == "route":
                if "_error" in raw:
                    print(f"[batch] route {qid}: {raw['_error']}", file=sys.stderr)
                    continue
                routes[qid] = {"selected_documents": selected_documents(raw, corpus, args), "reason": raw.get("reason")}
                continue
            selected = routes[qid]["selected_documents"]
            attempts[qid] = attempts.get(qid, 0) + 1
            if "_error" in raw:
                errs = [f"request failed: {raw['_error']}"]
                record = {}
            else:
                _, _, texts = draft_prompt(q, selected, corpus, None)
                record, errs = validate(raw, q, texts)
            status = "accepted" if not errs else ("failed" if attempts[qid] >= args.max_attempts else "retry")
            log.write(json.dumps({"id": qid, "status": status, "attempt": attempts[qid], "errors": errs,
                                  "selected_documents": selected, "route_reason": routes[qid].get("reason"),
                                  "model": args.model, "batch": True}, ensure_ascii=False) + "\n")
            log.flush()
            if not errs:
                write_draft(out, q, record, selected)
                done.add(qid)
                errors.pop(qid, None)
            else:
                errors[qid] = errs

    try:
        if state.get("job"):
            job = state["job"]
            print(f"[batch] resuming {job['kind']} job {job['name']}", file=sys.stderr)
            process(job["kind"], wait_batch(client, job["name"], args))
            state["job"] = None
            save()

        def run_jobs(kind: str, model: str, reqs: dict[str, dict[str, Any]]) -> None:
            """One job per token-budget group, one at a time (the cap counts active jobs)."""
            groups = split_by_tokens(reqs, args.batch_max_tokens)
            for i, group in enumerate(groups, start=1):
                label = f"gold_{kind}_{int(time.time())}_{i}of{len(groups)}"
                print(f"[batch] {kind} group {i}/{len(groups)}: {len(group)} requests, "
                      f"~{sum(estimated_tokens(r) for r in group.values()) / 1e6:.1f}M tokens", file=sys.stderr)
                name = submit_batch(client, model, group, label, args)
                state["job"] = {"name": name, "kind": kind}
                save()
                process(kind, wait_batch(client, name, args))
                state["job"] = None
                save()

        need_route = [q for q in todo if q["id"] not in routes]
        if need_route:
            reqs = {}
            for q in need_route:
                system, user, schema = route_prompt(q, corpus, topics, args)
                reqs[q["id"]] = batch_request(system, user, schema, args.temperature, args)
            run_jobs("route", args.route_model or args.model, reqs)

        while True:
            pending = [q for q in todo if q["id"] not in done and q["id"] in routes and attempts.get(q["id"], 0) < args.max_attempts]
            if not pending:
                break
            reqs = {}
            for q in pending:
                user, schema, _ = draft_prompt(q, routes[q["id"]]["selected_documents"], corpus, errors.get(q["id"]))
                reqs[q["id"]] = batch_request(DRAFT_SYSTEM, user, schema, args.temperature, args)
            run_jobs("draft", args.model, reqs)
            print(f"[batch] drafted {len(done & set(by_id))}/{len(by_id)}", file=sys.stderr)
    finally:
        out.close()
        log.close()


def draft(client: genai.Client, corpus: dict[str, list[dict[str, Any]]], topics: dict[str, str], args: argparse.Namespace) -> None:
    questions = [q for q in load_records(args.candidates) if q.get("id") and q.get("question")]
    done = {r["id"] for r in load_records(args.drafts)}
    todo = [q for q in questions if q["id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"[draft] {len(questions)} candidates, {len(done)} drafted, {len(todo)} to draft", file=sys.stderr)
    args.drafts.parent.mkdir(parents=True, exist_ok=True)
    if args.batch:
        draft_batch(corpus, topics, args, client, todo, done)
        return
    with args.drafts.open("a", encoding="utf-8") as out, args.log.open("a", encoding="utf-8") as log:
        for i, q in enumerate(todo, start=1):
            started = time.time()
            try:
                result = draft_one(client, q, corpus, topics, args)
            except Exception as e:
                result = {"id": q["id"], "status": "error", "error": str(e)[:500]}
            result.update(model=args.model, elapsed_seconds=round(time.time() - started, 1))
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()
            if result["status"] == "accepted":
                write_draft(out, q, result["record"], result["selected_documents"])
            print(f"[draft {i}/{len(todo)}] {q['id']} {result['status']} docs={result.get('selected_documents')} ({result['elapsed_seconds']}s)", file=sys.stderr)
            if args.sleep:
                time.sleep(args.sleep)


# ---------------------------------------------------------------- finalize


def finalize(corpus: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> None:
    drafts = load_records(args.drafts)
    n_ans = round(args.total * (1 - args.unanswerable_ratio))
    n_unans = args.total - n_ans
    rng = random.Random(args.seed)

    def matches_target(d: dict[str, Any]) -> bool:
        return bool(set(d.get("target_passages") or []) & set(d.get("gold_passages") or []))

    answerable = [d for d in drafts if d.get("intended") == "answerable" and d["answerable"] in ("yes", "partial")]
    unanswerable = [d for d in drafts if d.get("intended") == "unanswerable" and d["answerable"] == "no"]
    rank = lambda d: (not matches_target(d), d["answerable"] != "yes", d["confidence"] == "low", d["id"])

    quotas = allocate(n_ans, {d: len(ps) for d, ps in corpus.items()}, args.min_per_document)
    by_doc: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for d in sorted(answerable, key=rank):
        by_doc[d.get("source_document")].append(d)
    chosen: list[dict[str, Any]] = []
    for doc, quota in quotas.items():
        chosen += by_doc.get(doc, [])[:quota]
    chosen_ids = {d["id"] for d in chosen}
    chosen += [d for d in sorted(answerable, key=rank) if d["id"] not in chosen_ids][: n_ans - len(chosen)]

    def is_nao(d: dict[str, Any]) -> bool:
        return d["question_type"] == "binary" and d.get("short_answer") == "Não"

    def nao_share(rows: list[dict[str, Any]]) -> float:
        binary = [d for d in rows if d["question_type"] == "binary" and d.get("short_answer")]
        return sum(is_nao(d) for d in binary) / len(binary) if binary else 0.0

    # Swap the weakest "Não" binaries for the best unused non-"Não" candidates
    # (same resolution first) until the "Não" share is at most --max-nao-share.
    pool = [d for d in sorted(answerable, key=rank) if d["id"] not in {c["id"] for c in chosen}]
    swaps = 0
    while nao_share(chosen) > args.max_nao_share:
        naos = [d for d in chosen if is_nao(d)]
        if not naos:
            break
        out = max(naos, key=rank)
        repl = next((d for d in pool if not is_nao(d) and d.get("source_document") == out.get("source_document")), None)
        repl = repl or next((d for d in pool if not is_nao(d)), None)
        if repl is None:
            break
        chosen.remove(out)
        chosen.append(repl)
        pool.remove(repl)
        swaps += 1

    by_kind = {k: sorted((d for d in unanswerable if d.get("unanswerable_kind") == k), key=lambda d: d["id"]) for k in UNANSWERABLE_KINDS}
    other = sorted((d for d in unanswerable if d.get("unanswerable_kind") not in UNANSWERABLE_KINDS), key=lambda d: d["id"])
    picked_unans = by_kind["near_miss"][: math.ceil(n_unans / 2)] + by_kind["out_of_scope"][: n_unans // 2]
    picked_ids = {d["id"] for d in picked_unans}
    picked_unans += [d for d in by_kind["near_miss"] + by_kind["out_of_scope"] + other if d["id"] not in picked_ids][: n_unans - len(picked_unans)]
    chosen += picked_unans
    rng.shuffle(chosen)

    def g_is_no(d: dict[str, Any]) -> bool:
        return d["answerable"] == "no"

    gold: list[dict[str, Any]] = []
    for n, d in enumerate(chosen, start=1):
        gold.append({
            **{f: d.get(f) for f in GOLD_FIELDS},
            "id": f"q{n:03d}",
            "candidate_id": d["id"],
            "source_document": d.get("source_document"),
            "unanswerable_kind": d.get("unanswerable_kind") if g_is_no(d) else None,
        })
    write_jsonl(args.output, gold)
    args.questions_out.write_text(
        json.dumps([{"id": g["id"], "question": g["question"]} for g in gold], ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    starts = collections.Counter(" ".join(g["question"].split()[:2]) for g in gold)
    review = sorted(
        g["id"] for g, d in zip(gold, chosen)
        if g["answerable"] != "yes" or g["confidence"] == "low" or (d.get("intended") == "answerable" and not matches_target(d))
    )
    report = {
        "model": args.model,
        "target": {"total": args.total, "answerable": n_ans, "unanswerable": n_unans},
        "selected": len(gold),
        "shortfall": {"answerable": max(0, n_ans - sum(g["answerable"] != "no" for g in gold)),
                      "unanswerable": max(0, n_unans - sum(g["answerable"] == "no" for g in gold))},
        "candidates_drafted": len(drafts),
        "drafter_disagreed_with_intent": {
            "intended_unanswerable_but_answerable": sum(1 for d in drafts if d.get("intended") == "unanswerable" and d["answerable"] != "no"),
            "intended_unanswerable_but_answerable_by_kind": dict(collections.Counter(
                d.get("unanswerable_kind") for d in drafts if d.get("intended") == "unanswerable" and d["answerable"] != "no"
            )),
            "intended_answerable_but_no": sum(1 for d in drafts if d.get("intended") == "answerable" and d["answerable"] == "no"),
            "answerable_gold_outside_source_passages": sum(1 for d in answerable if not matches_target(d)),
        },
        "answerable": dict(collections.Counter(g["answerable"] for g in gold)),
        "question_type": dict(collections.Counter(g["question_type"] for g in gold)),
        "short_answer": dict(collections.Counter(g["short_answer"] for g in gold if g["question_type"] == "binary")),
        "binary_nao_share": round(nao_share(chosen), 3),
        "nao_swaps": swaps,
        "unanswerable_kind": dict(collections.Counter(g["unanswerable_kind"] for g in gold if g["answerable"] == "no")),
        "confidence": dict(collections.Counter(g["confidence"] for g in gold)),
        "by_source_document": dict(collections.Counter(g["source_document"] or "unanswerable" for g in gold)),
        "top_question_starts": starts.most_common(5),
        "review_first": review,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("selected", "shortfall", "answerable", "question_type", "short_answer", "binary_nao_share", "unanswerable_kind")}, ensure_ascii=False))
    print(f"-> {args.output}\n-> {args.questions_out}\n-> {args.report} ({len(review)} records to review first)", file=sys.stderr)
    if any(report["shortfall"].values()):
        print("shortfall: raise --overgenerate / --unanswerable-overgenerate and rerun generate with --force, or review failed drafts in the log", file=sys.stderr)


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description="Fresh evaluation set (questions + gold) drafted with Gemini.")
    ap.add_argument("--stage", choices=["generate", "draft", "finalize", "all"], default="all")
    ap.add_argument("--total", type=int, default=200)
    ap.add_argument("--unanswerable-ratio", type=float, default=0.10)
    ap.add_argument("--overgenerate", type=float, default=1.2, help="answerable candidates per final slot")
    ap.add_argument("--unanswerable-overgenerate", type=float, default=2.5, help="unanswerable candidates per final slot (near-misses are rejected more often)")
    ap.add_argument("--max-nao-share", type=float, default=0.6, help="max share of 'Não' among binary short answers in the final set")
    ap.add_argument("--min-per-document", type=int, default=5)
    ap.add_argument("--per-call", type=int, default=12, help="questions requested per generation call")
    ap.add_argument("--similarity-threshold", type=float, default=0.6, help="word-set Jaccard to drop a question")
    ap.add_argument("--passages", type=Path, default=Path("../data_pipeline/data/passages.jsonl"))
    ap.add_argument("--training-qa", type=Path, default=Path("../data_pipeline/data/qa_pairs_final.jsonl"))
    ap.add_argument("--previous-questions", type=Path, default=None, help="optional earlier question set whose questions must not be repeated")
    ap.add_argument("--candidates", type=Path, default=Path("questions/candidates.jsonl"))
    ap.add_argument("--generation-log", type=Path, default=Path("questions/candidates_raw.jsonl"),
                    help="per-chunk generation checkpoint; a rerun reuses finished chunks")
    ap.add_argument("--drafts", type=Path, default=Path("questions/gold_drafts.jsonl"))
    ap.add_argument("--log", type=Path, default=Path("questions/gold_log.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("questions/gold.jsonl"))
    ap.add_argument("--questions-out", type=Path, default=Path("questions/questions.json"))
    ap.add_argument("--report", type=Path, default=Path("questions/gold_report.json"))
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--route-model", default="gemini-3.8-flash", help="model for the routing step ('' = same as --model)")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--api-version", default="v1beta", help="Gemini API version; preview models are served on v1beta")
    ap.add_argument("--list-models", action="store_true", help="print the models this key can call with generateContent, then exit")
    ap.add_argument("--temperature", type=float, default=0.0, help="draft/route temperature")
    ap.add_argument("--gen-temperature", type=float, default=0.9, help="question-generation temperature")
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--max-documents", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=2, help="drafts per question when validation fails")
    ap.add_argument("--max-api-retries", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=10.0, help="first wait (s) after a rate-limit/server error")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--sleep", type=float, default=0.0, help="pause (s) between drafted questions")
    ap.add_argument("--limit", type=int, default=None, help="draft at most N candidates this run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="regenerate candidates even if the file exists")
    ap.add_argument("--batch", action="store_true", help="draft stage through the Batch API (half price, asynchronous)")
    ap.add_argument("--batch-dir", type=Path, default=Path("questions/batch"), help="batch request files")
    ap.add_argument("--batch-state", type=Path, default=Path("questions/gold_batch_state.json"), help="routes, attempts and pending job")
    ap.add_argument("--poll-seconds", type=float, default=60.0, help="wait between batch status checks")
    ap.add_argument("--batch-max-tokens", type=int, default=4_000_000,
                    help="estimated input tokens per batch job; jobs run one at a time (Tier 1 cap: 5M active per model)")
    args = ap.parse_args()

    client = None
    if args.list_models or args.stage in ("generate", "draft", "all"):
        key = os.environ.get(args.api_key_env)
        if not key:
            raise SystemExit(f"missing env {args.api_key_env}")
        client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(api_version=args.api_version, timeout=args.timeout * 1000),
        )
    if args.list_models:
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                print(m.name, "batch" if "batchGenerateContent" in actions else "")
        return
    corpus, topics = load_corpus(args.passages)
    if args.stage in ("generate", "all"):
        generate(client, corpus, topics, args)
    if args.stage in ("draft", "all"):
        draft(client, corpus, topics, args)
    if args.stage in ("finalize", "all"):
        finalize(corpus, args)


if __name__ == "__main__":
    main()
