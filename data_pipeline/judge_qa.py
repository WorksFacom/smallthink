#!/usr/bin/env python3
"""
Dual-judge quality gate for CFM Q&A pairs (pt-BR), using two local SLMs
served by Ollama (OpenAI-compatible chat-completions endpoint at /v1).

What it does
------------
Reads the pairs produced by generate_qa.py (data/qa_pairs.jsonl), where each
record carries its source passage in `context`. Each pair is judged TWICE --
once by each of two *different* SLMs (default: Gemma 4 and qwen3.5:9b) --
against its own `context`. Each judge returns one of:

  - "boa"            : the answer correctly/usefully answers the question.
  - "corrigivel"     : the answer is flawed (e.g. it just restates the
                       question, is vague, or is incomplete) BUT the context
                       has enough to write a correct answer; the judge then
                       supplies a grounded `resposta_corrigida`.
  - "irrecuperavel"  : the context does NOT contain what the question asks
                       (e.g. a "quais sao" question over a passage that never
                       enumerates), so it cannot be fixed without inventing.

Reconciliation (keep-biased, fully automatic -- no manual step)
---------------------------------------------------------------
A pair is DISCARDED only when BOTH judges call it "irrecuperavel". On any
other outcome the pair is KEPT; if at least one judge marked it "corrigivel",
its grounded correction is applied (judge A's preferred). This biases the gate
against throwing away salvageable pairs -- which matters on a small corpus.
The continuous 0-5 score from each judge is recorded (NOT used for the cut by
default) so you can inspect the distribution afterwards.

Why two sequential passes (not interleaved): so Ollama doesn't swap models in
and out of VRAM on every item. All pairs go through judge A, then all through
judge B; each model loads once per run.

Grounding discipline: corrections must use ONLY the passage -- no external
knowledge -- which matters in a medical/legal domain. An errored or
unparseable judge response can never cause a discard (fail-safe toward KEEP).

Usage
-----
  python judge_qa.py \
      --input data/qa_pairs.jsonl \
      --output data/qa_pairs_reviewed.jsonl \
      --clean-output data/qa_pairs_clean.jsonl \
      --judge-model-a gemma4 \
      --judge-model-b qwen3.5:9b

Model names are passed to Ollama verbatim -- match them to `ollama list`.
The run is resumable: pairs already present in --output are skipped, so you can
re-run it after generating more pairs.
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

JUDGE_PROMPT_VERSION = "cfm-judge-v2"
VALID_CATEGORIES = ("boa", "corrigivel", "irrecuperavel")

JUDGE_SYSTEM_PROMPT = (
    "Voce e um avaliador rigoroso de pares de pergunta-resposta sobre etica "
    "medica e direito medico brasileiro. Cada par foi gerado a partir de um "
    "TRECHO de uma norma do Conselho Federal de Medicina (CFM), que tambem e "
    "fornecido. Sua tarefa e julgar se a RESPOSTA realmente responde a "
    "PERGUNTA, usando SOMENTE o que esta no TRECHO.\n\n"
    "Classifique o par em UMA categoria:\n"
    "- \"boa\": a resposta responde a pergunta de forma correta, objetiva e "
    "informativa, com base no TRECHO.\n"
    "- \"corrigivel\": a resposta tem defeito (por exemplo: apenas repete ou "
    "reformula a pergunta sem informar nada novo; e vaga; esta incompleta), "
    "MAS o TRECHO contem informacao suficiente para escrever uma resposta "
    "correta. Nesse caso, forneca \"resposta_corrigida\" baseada SOMENTE no "
    "TRECHO.\n"
    "- \"irrecuperavel\": o TRECHO NAO contem a informacao que a pergunta "
    "pede, entao nao ha como corrigir sem inventar. Exemplo tipico: a pergunta "
    "pede \"quais sao\" ou \"liste\" mas o TRECHO apenas menciona o conceito "
    "sem enumerar os itens. NAO forneca resposta_corrigida.\n\n"
    "Regras:\n"
    "1. Use SOMENTE o TRECHO. Nunca use conhecimento externo nem complete com "
    "suposicoes. Se a informacao necessaria nao esta no TRECHO, a categoria e "
    "OBRIGATORIAMENTE \"irrecuperavel\".\n"
    "2. A resposta_corrigida deve ser autocontida, em portugues do Brasil, e "
    "terminar com uma linha no formato: Fonte: <citation>, usando exatamente "
    "a citacao canonica informada.\n"
    "3. Atribua um score INTEIRO de 0 a 5 para a qualidade da resposta "
    "ORIGINAL (0 = nao responde / irrecuperavel; 5 = excelente).\n"
    "4. Seja conservador ao classificar como \"irrecuperavel\": so use essa "
    "categoria quando tiver certeza de que o TRECHO nao permite responder.\n"
    "5. Saida ESTRITA em JSON valido, sem nenhum texto fora do JSON, no "
    "formato: {\"categoria\": \"boa|corrigivel|irrecuperavel\", \"score\": 0, "
    "\"motivo\": \"...\", \"resposta_corrigida\": \"\"}."
)

JUDGE_USER_TEMPLATE = (
    "Avalie o par a seguir.\n\n"
    "Citacao canonica: {citation}\n"
    "Paginas: {page_start}-{page_end}\n\n"
    "TRECHO:\n\"\"\"\n{context}\n\"\"\"\n\n"
    "PERGUNTA:\n\"\"\"\n{question}\n\"\"\"\n\n"
    "RESPOSTA:\n\"\"\"\n{answer}\n\"\"\"\n\n"
    "Responda SOMENTE com JSON no formato indicado."
)


# --------------------------------------------------------------------------
# Ollama HTTP
# --------------------------------------------------------------------------
def _call_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    json_mode: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

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


def _extract_json(raw: str) -> str:
    """Pull a JSON object out of a model reply, tolerating reasoning models.

    Strips <think>/<thinking> blocks and code fences, then takes the outermost
    {...} span. Robust enough for both plain-instruct and thinking SLMs.
    """
    s = raw.strip()
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<thinking>[\s\S]*?</thinking>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```(?:json)?", "", s)
    s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


# --------------------------------------------------------------------------
# Verdict parsing
# --------------------------------------------------------------------------
def _normalize_verdict(obj: dict[str, Any]) -> dict[str, Any]:
    cat = str(obj.get("categoria", "")).strip().lower()
    if cat not in VALID_CATEGORIES:
        # Unknown category -> keep-biased default; never able to force a discard.
        cat = "boa"

    score = obj.get("score", None)
    try:
        score = int(round(float(score)))
        score = max(0, min(5, score))
    except (TypeError, ValueError):
        score = None

    motivo = str(obj.get("motivo", "")).strip()
    corr = str(obj.get("resposta_corrigida", "") or "").strip()
    if cat != "corrigivel":
        corr = ""  # corrections only make sense for the fixable category

    return {"categoria": cat, "score": score, "motivo": motivo, "resposta_corrigida": corr}


def _error_verdict(msg: str) -> dict[str, Any]:
    # "erro" is treated as NOT irrecoverable, so an errored judge cannot
    # contribute to a discard (fail-safe toward keeping the pair).
    return {"categoria": "erro", "score": None, "motivo": msg, "resposta_corrigida": ""}


def _judge_one(
    rec: dict[str, Any],
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    json_mode: bool,
) -> dict[str, Any]:
    user = JUDGE_USER_TEMPLATE.format(
        citation=rec.get("citation", "?"),
        page_start=rec.get("page_start", "?"),
        page_end=rec.get("page_end", "?"),
        context=rec.get("context", ""),
        question=rec.get("question", ""),
        answer=rec.get("answer", ""),
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        resp = _call_chat(base_url, model, messages, temperature, max_tokens, timeout, json_mode)
        content = _extract_content(resp)
        obj = json.loads(_extract_json(content))
        if not isinstance(obj, dict):
            raise ValueError("verdict is not a JSON object")
        return _normalize_verdict(obj)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as e:
        return _error_verdict(f"{type(e).__name__}: {e}")


def run_pass(
    records: list[dict[str, Any]],
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    json_mode: bool,
    concurrency: int,
    label: str,
) -> dict[str, dict[str, Any]]:
    """Judge every record with a single model. Returns {qa_id: verdict}."""
    verdicts: dict[str, dict[str, Any]] = {}
    total = len(records)
    done = 0
    errors = 0
    start = time.time()
    with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        fut_to_id = {
            ex.submit(
                _judge_one, rec, model, base_url, temperature, max_tokens, timeout, json_mode
            ): rec["qa_id"]
            for rec in records
        }
        for fut in cf.as_completed(fut_to_id):
            qa_id = fut_to_id[fut]
            v = fut.result()
            verdicts[qa_id] = v
            done += 1
            if v["categoria"] == "erro":
                errors += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - start
                print(
                    f"[juiz {label}] {done}/{total} avaliados (erros={errors}) "
                    f"elapsed={elapsed:0.1f}s",
                    file=sys.stderr,
                )
    return verdicts


# --------------------------------------------------------------------------
# Reconciliation (keep-biased)
# --------------------------------------------------------------------------
def reconcile(
    va: dict[str, Any],
    vb: dict[str, Any],
    original_answer: str,
) -> tuple[str, str, bool]:
    """Return (review_status, final_answer, was_corrected)."""
    ca, cb = va["categoria"], vb["categoria"]

    # Discard ONLY when both judges agree the pair is unanswerable from source.
    if ca == "irrecuperavel" and cb == "irrecuperavel":
        return "auto_discarded", original_answer, False

    # Otherwise keep; apply a grounded correction if any judge offered one
    # (judge A preferred).
    if ca == "corrigivel" and va["resposta_corrigida"]:
        return "auto_corrected", va["resposta_corrigida"], True
    if cb == "corrigivel" and vb["resposta_corrigida"]:
        return "auto_corrected", vb["resposta_corrigida"], True

    return "auto_kept", original_answer, False


# --------------------------------------------------------------------------
# IO / resume / reporting
# --------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("qa_id")
            if qid:
                ids.add(qid)
    return ids


def _print_score_histogram(reviewed: list[dict[str, Any]]) -> None:
    buckets = {i: 0 for i in range(6)}
    nonum = 0
    for r in reviewed:
        judges = r.get("judges", {})
        scores = [
            judges.get(k, {}).get("score")
            for k in ("a", "b")
            if isinstance(judges.get(k, {}).get("score"), int)
        ]
        if not scores:
            nonum += 1
            continue
        avg = int(round(sum(scores) / len(scores)))
        buckets[max(0, min(5, avg))] += 1

    print("\nDistribuicao de score (media dos 2 juizes):", file=sys.stderr)
    width = 40
    mx = max(buckets.values()) or 1
    for i in range(6):
        bar = "#" * int(width * buckets[i] / mx)
        print(f"  score {i}: {buckets[i]:4d} {bar}", file=sys.stderr)
    if nonum:
        print(f"  (sem score numerico: {nonum})", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dual-SLM quality gate for CFM Q&A pairs via a local Ollama server.",
    )
    ap.add_argument("--input", type=Path, default=Path("data/qa_pairs.jsonl"))
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/qa_pairs_reviewed.jsonl"),
        help="All pairs + verdicts + status (full audit trail).",
    )
    ap.add_argument(
        "--clean-output",
        type=Path,
        default=None,
        help="If set, also writes ONLY survivors (kept/corrected) with the final answer.",
    )
    ap.add_argument(
        "--judge-model-a",
        default="gemma4",
        help="Primary judge (Ollama tag). Default: gemma4. Match to `ollama list`.",
    )
    ap.add_argument(
        "--judge-model-b",
        default="qwen3.5:9b",
        help="Secondary judge (Ollama tag). Default: qwen3.5:9b. Match to `ollama list`.",
    )
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Raise if a reasoning judge truncates its JSON (thinking eats tokens).",
    )
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="Max pairs to process.")
    ap.add_argument(
        "--json-mode",
        action="store_true",
        help="Send response_format=json_object. Off by default for broad model compatibility.",
    )
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip pairs already in --output (default). Use --no-resume to start fresh.",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    records = _load_jsonl(args.input)
    records = [
        r for r in records
        if r.get("qa_id") and r.get("question") and r.get("answer") is not None
    ]
    if args.limit is not None:
        records = records[: args.limit]

    if args.resume:
        done_ids = _existing_ids(args.output)
        records = [r for r in records if r["qa_id"] not in done_ids]

    if not records:
        print("Nada a fazer (todos os pares ja avaliados?).", file=sys.stderr)
        # Still (re)build clean-output / report from whatever is on disk.
        if args.output.is_file():
            all_reviewed = _load_jsonl(args.output)
            _emit_clean_and_report(all_reviewed, args.clean_output, args.output)
        return

    print(
        f"Avaliando {len(records)} par(es) com A={args.judge_model_a!r} e depois "
        f"B={args.judge_model_b!r}...",
        file=sys.stderr,
    )

    # Two sequential passes -> each model loads into VRAM once.
    verdicts_a = run_pass(
        records, args.judge_model_a, args.base_url, args.temperature,
        args.max_tokens, args.timeout, args.json_mode, args.concurrency, label="A",
    )
    verdicts_b = run_pass(
        records, args.judge_model_b, args.base_url, args.temperature,
        args.max_tokens, args.timeout, args.json_mode, args.concurrency, label="B",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and args.output.exists()) else "w"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with args.output.open(mode, encoding="utf-8") as out_f:
        for rec in records:
            qid = rec["qa_id"]
            va = verdicts_a.get(qid, _error_verdict("verdict A ausente"))
            vb = verdicts_b.get(qid, _error_verdict("verdict B ausente"))
            original = rec["answer"]
            status, final_answer, _ = reconcile(va, vb, original)

            new_rec = dict(rec)
            new_rec["answer_original"] = original
            new_rec["answer"] = final_answer
            new_rec["review_status"] = status
            new_rec["judge_agreement"] = va["categoria"] == vb["categoria"]
            new_rec["judges"] = {
                "a": {"model": args.judge_model_a, **va},
                "b": {"model": args.judge_model_b, **vb},
            }
            new_rec["judge_prompt_version"] = JUDGE_PROMPT_VERSION
            new_rec["judged_at"] = now

            out_f.write(json.dumps(new_rec, ensure_ascii=False))
            out_f.write("\n")
        out_f.flush()

    # Recompute report + clean-output over the FULL file (correct under --resume).
    all_reviewed = _load_jsonl(args.output)
    _emit_clean_and_report(all_reviewed, args.clean_output, args.output)


def _emit_clean_and_report(
    all_reviewed: list[dict[str, Any]],
    clean_output: Path | None,
    output: Path,
) -> None:
    if clean_output is not None:
        clean_output.parent.mkdir(parents=True, exist_ok=True)
        with clean_output.open("w", encoding="utf-8") as cf_:
            for r in all_reviewed:
                if r.get("review_status") != "auto_discarded":
                    cf_.write(json.dumps(r, ensure_ascii=False))
                    cf_.write("\n")

    _print_score_histogram(all_reviewed)

    kept = sum(1 for r in all_reviewed if r.get("review_status") == "auto_kept")
    corrected = sum(1 for r in all_reviewed if r.get("review_status") == "auto_corrected")
    discarded = sum(1 for r in all_reviewed if r.get("review_status") == "auto_discarded")
    agree = sum(1 for r in all_reviewed if r.get("judge_agreement"))
    total = len(all_reviewed)
    survivors = kept + corrected
    pct = (100 * survivors / total) if total else 0.0

    print(
        f"\nPronto. avaliados={total} | mantidos={kept} corrigidos={corrected} "
        f"descartados={discarded} | sobreviventes={survivors} ({pct:0.1f}%) "
        f"| concordancia={agree}/{total}",
        file=sys.stderr,
    )
    msg = f"-> revisado: {output}"
    if clean_output is not None:
        msg += f"\n-> limpo:    {clean_output}"
    print(msg)


if __name__ == "__main__":
    main()
