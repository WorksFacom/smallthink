#!/usr/bin/env python3
"""
Ingest CFM (Conselho Federal de Medicina) legal PDFs into a QA-ready dataset.

Outputs (default under ./data/):
  data/extracted/<document_id>.json   one structured JSON per PDF
  data/passages.jsonl                 flat passages across all docs
  data/manifest.json                  catalog of documents

Flat passage kinds emitted: ementa, considerando, artigo, item, assinatura,
exposicao_motivos.

Structure handled beyond plain "Art. N" splitting:
  - Chapter headings ("Capítulo IX / SIGILO PROFISSIONAL / É vedado ao médico:")
    end the preceding article and are prepended to each article below them as
    ``context``, so "Art. 73 Revelar fato..." reads as a prohibition.
  - Roman-numeral items in chapters without articles (Código de Ética Médica:
    Preâmbulo, Princípios Fundamentais, Direitos dos Médicos, Disposições
    Gerais) become one ``item`` passage each.
  - Articles longer than LONG_ARTICLE_CHARS are split at incisos/parágrafos
    into ``<id>_pN`` parts that repeat the caput.
  - Page header/footer lines (address, portal URL) are removed.
  - Identical passages across documents get ``duplicate_of`` (first seen
    wins) and the kept one lists the other citations in ``also_cited_as``.

Usage:
  python3 ingest_cfm.py --input-dir legal_documents --output-dir data
  python3 ingest_cfm.py legal_documents/2217_2018-CEM.pdf --output-dir data
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _normalize_text(text: str, form: str | None) -> str:
    if not form:
        return text
    return unicodedata.normalize(form, text)


def _normalize_pages(pages: list[dict[str, Any]], form: str | None) -> None:
    for p in pages:
        if "text" in p and isinstance(p["text"], str):
            p["text"] = _normalize_text(p["text"], form)


def _normalize_metadata_strings(meta: dict[str, Any], form: str | None) -> None:
    if not form:
        return
    for key in ("title", "author", "subject"):
        v = meta.get(key)
        if isinstance(v, str) and v:
            meta[key] = _normalize_text(v, form)


def _read_pdf(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    pages: list[dict[str, Any]] = []
    try:
        meta = doc.metadata or {}
        for i in range(len(doc)):
            page = doc[i]
            pages.append({"page": i + 1, "text": page.get_text()})
        extra: dict[str, Any] = {
            "page_count": len(doc),
            "title": meta.get("title") or None,
            "author": meta.get("author") or None,
            "subject": meta.get("subject") or None,
        }
    finally:
        doc.close()
    return pages, extra


def _read_text_file(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return [{"page": 1, "text": raw}]


def _full_text(pages: list[dict[str, Any]], separator: str = "\n\n") -> str:
    return separator.join(p["text"] for p in pages if p.get("text"))


def _prepare_document(
    path: Path,
    unicode_form: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages, pdf_meta = _read_pdf(path)
        fmt = "pdf"
        metadata = pdf_meta
    elif suffix in {".txt", ".md"}:
        pages = _read_text_file(path)
        fmt = "text"
        metadata = {"page_count": len(pages)}
    else:
        raise ValueError(f"Unsupported format {suffix!r}. Use .pdf, .txt, or .md.")
    _normalize_pages(pages, unicode_form)
    _normalize_metadata_strings(metadata, unicode_form)
    full = _full_text(pages)
    return pages, metadata, fmt, full


def _page_char_spans(
    pages: list[dict[str, Any]],
    sep: str = "\n\n",
) -> list[tuple[int, int, int]]:
    """Character ranges [start, end) in concatenated full text -> PDF page number."""
    spans: list[tuple[int, int, int]] = []
    pos = 0
    for i, p in enumerate(pages):
        t = p.get("text") or ""
        start = pos
        end = pos + len(t)
        spans.append((start, end, int(p["page"])))
        pos = end + (len(sep) if i < len(pages) - 1 else 0)
    return spans


def _span_pages(
    char_start: int,
    char_end: int,
    spans: list[tuple[int, int, int]],
) -> tuple[int, int]:
    """Smallest page range overlapping [char_start, char_end)."""
    pages_hit = [pg for (s, e, pg) in spans if e > char_start and s < char_end]
    if not pages_hit:
        return 1, 1
    return min(pages_hit), max(pages_hit)


SCHEMA = "cfm_document"
SCHEMA_VERSION = "1.1"
DOMAIN = "brazil_medical_ethics_legal"
LANGUAGE = "pt-BR"

RE_ACT_HEADER = re.compile(
    r"RESOLU[\u00c7C][A\u00c3]O\s+CFM\s+N[\u00ba\u00b0oO\.]*\s*([\d\.]+)\s*/\s*(\d{4})",
    re.IGNORECASE,
)
RE_PUBLICATION_EM = re.compile(
    r"Publicad[oa]\s+em[:\s]+([^\n]+)",
    re.IGNORECASE,
)
RE_DOU_LINE = re.compile(
    r"\(?\s*Publicad[oa]\s+no\s+D\.O\.U\.[^\n)]*\)?",
    re.IGNORECASE,
)
RE_COUNCIL_START = re.compile(
    r"\bO\s+CONSELHO\s+FEDERAL\s+DE\s+MEDICINA\b",
    re.IGNORECASE,
)
RE_CONSIDERANDO = re.compile(r"\bCONSIDERANDO\b", re.IGNORECASE)
RE_RESOLVE = re.compile(r"\bRESOLVE\s*:?", re.IGNORECASE)
RE_ART_HEAD = re.compile(
    r"(?m)^\s*(Art(?:igo)?\.?)\s*(\d+)[\u00ba\u00b0oO\.]*",
    re.IGNORECASE,
)
RE_MODIFICADA = re.compile(
    r"MODIFICAD[AO]\s*:\s*\n((?:\s*Resolu[\u00e7c][\u00e3a]o\s+CFM\s+n[\u00ba\u00b0o]\s*[\d\./]+\s*\n?)+)",
    re.IGNORECASE,
)
RE_REVOGA_EXPLICIT = re.compile(
    r"revoga[\u00e7c][\u00e3a]o\s+d[ao]\s+(Resolu[\u00e7c][\u00e3a]o\s+CFM\s+n[\u00ba\u00b0o]\s*[\d\./]+)",
    re.IGNORECASE,
)
RE_REVOGA_LIST = re.compile(
    r"Ficam?\s+revogad[ao]s?\s+as\s+Resolu[\u00e7c][\u00f5o]es\s+CFM\s+n[\u00ba\u00b0o]s?\s*([^\.\n]+)",
    re.IGNORECASE,
)
RE_DATE_SIGNED = re.compile(
    r"Bras[\u00edi]lia[,\s\-]*(?:DF)?[,\s\-]*(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})",
    re.IGNORECASE,
)
RE_SIG_ROLE_INLINE = re.compile(
    r"(?i)\b(Presidente(?:\s+do\s+CFM)?|Relator(?:a)?|"
    r"Secret[\u00e1a]rio[\- ]?[Gg]eral|Secret[\u00e1a]rio-[Gg]eral|"
    r"Conselheir[ao](?:\s+relator[ao]?)?)\b"
)
RE_NAME_LINE = re.compile(
    r"^[A-Z\u00c1\u00c0\u00c2\u00c3\u00c9\u00ca\u00cd\u00d3\u00d4\u00d5\u00da\u00c7]"
    r"[A-Z\u00c1\u00c0\u00c2\u00c3\u00c9\u00ca\u00cd\u00d3\u00d4\u00d5\u00da\u00c7\s\.\-']{3,}$"
)
RE_EXPOSICAO = re.compile(
    r"\bEXPOSI[\u00c7C][A\u00c3]O\s+DE\s+MOTIVOS\b",
    re.IGNORECASE,
)
RE_CHAPTER = re.compile(
    r"(?m)^[ \t]*(?:CAP[\u00cdI]TULO|Cap[\u00edi]tulo)[ \t]+([IVXLC]+)[ \t]*$"
)
RE_PREAMBULO = re.compile(r"(?m)^[ \t]*PRE[\u00c2A]MBULO[ \t]*$")
RE_NEXT_LINE = re.compile(r"\n[ \t]*([^\n]*)")
RE_LEAD_IN = re.compile(r"^\u00c9\s+(?:vedado|direito|dever)\b.*:$", re.IGNORECASE)
RE_ITEM_HEAD = re.compile(r"(?m)^[ \t]*([IVXLC]+)[ \t]*[\u2013\-][ \t]*")
RE_SUBDIVISION = re.compile(
    r"(?m)^[ \t]*(?:[IVXLC]+[ \t]*[\u2013\-]|[IVXLC]+\.(?:[ \t]|$)|\u00a7|Par[\u00e1a]grafo\s+[\u00fau]nico)"
)
RE_NOISE_LINE = re.compile(
    r"(?im)^[ \t]*(?:.*\bSGAS\s+915\b.*|.*\bCEP:\s*70390.*|.*portalmedico\.org\.br.*"
    r"|CONSELHO FEDERAL DE MEDICINA|\d{1,3})[ \t]*$"
)
# Above this many characters an article is split at incisos/parágrafos;
# bge-m3 vectors of multi-topic articles match no single question well.
LONG_ARTICLE_CHARS = 1500
PART_CHARS = 1000
DEDUPE_KINDS = {"ementa", "considerando", "artigo", "item"}

_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "mar\u00e7o": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
    "outubro": 10, "novembro": 11, "dezembro": 12,
}


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(s: str) -> str:
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _squeeze(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _body(raw: str) -> str:
    return _clean(RE_NOISE_LINE.sub("", raw))


def _with_context(context: str | None, body: str) -> str:
    return f"{context}\n{body}" if context else body


def _canonical_document_id(number: str, year: int) -> str:
    return f"cfm_res_{number}_{year}"


def _find_act(full: str, filename: str) -> tuple[str, int, str]:
    m = RE_ACT_HEADER.search(full)
    if m:
        return m.group(1).replace(".", ""), int(m.group(2)), m.group(0)
    stem = Path(filename).stem
    m2 = re.search(r"(\d{3,5})[_\-\s]+(?:de\s+)?(\d{4})", stem)
    if m2:
        return m2.group(1), int(m2.group(2)), ""
    return "unknown", 0, ""


def _extract_ementa(full: str) -> tuple[int, int, str] | None:
    cm = RE_COUNCIL_START.search(full)
    if not cm:
        return None
    hm = RE_ACT_HEADER.search(full)
    start_scan = hm.end() if hm else 0
    region = full[start_scan:cm.start()]
    paragraphs = re.split(r"\n\s*\n", region)
    offset = start_scan
    for p in paragraphs:
        stripped = p.strip()
        if stripped:
            low = stripped.lower()
            skip = (
                "sgas" in low
                or "cep:" in low
                or "fone:" in low
                or "fax:" in low
                or "portalmedico" in low
                or "publicad" in low
                or "d.o.u" in low
                or "conselho federal de medicina" == stripped.lower()
                or "aten\u00e7\u00e3o ao apostilamento" in low
                or low.startswith("modificad")
            )
            if not skip and len(stripped) >= 30:
                s = full.find(p, offset)
                if s < 0:
                    s = offset
                return s, s + len(p), _squeeze(stripped)
        offset += len(p) + 2
    return None


def _extract_publication(full: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = RE_PUBLICATION_EM.search(full)
    if m:
        raw = m.group(0).strip()
        out["raw"] = raw
        m3 = re.search(
            r"(\d{1,2}/\d{1,2}/\d{4})"
            r"(?:.*?Edi[\u00e7c][\u00e3a]o:\s*([^\|\n]+?))?"
            r"(?:.*?Se[\u00e7c][\u00e3a]o:\s*([^\|\n]+?))?"
            r"(?:.*?P[\u00e1a]gina:\s*([^\|\n]+))?",
            raw,
        )
        if m3:
            out["date"] = m3.group(1)
            out["journal"] = "DOU"
            if m3.group(2):
                out["edition"] = m3.group(2).strip()
            if m3.group(3):
                out["section"] = m3.group(3).strip()
            if m3.group(4):
                out["page"] = m3.group(4).strip()
        return out
    m2 = RE_DOU_LINE.search(full)
    if m2:
        out["raw"] = m2.group(0).strip(" ()")
        out["journal"] = "DOU"
    return out


def _extract_considerandos(full: str, body_end: int) -> list[tuple[int, int, str]]:
    starts = [m.start() for m in RE_CONSIDERANDO.finditer(full) if m.start() < body_end]
    out: list[tuple[int, int, str]] = []
    for i, s in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else body_end
        cut = min(nxt, body_end)
        art = RE_ART_HEAD.search(full, s + 1, cut)
        if art:
            cut = min(cut, art.start())
        res = RE_RESOLVE.search(full, s + 1, cut)
        if res:
            cut = min(cut, res.start())
        text = _body(full[s:cut])
        if text:
            out.append((s, cut, text))
    return out


def _extract_headings(full: str) -> list[dict[str, Any]]:
    """Chapter/preamble headings with their title lines and lead-in phrase."""
    found = [(m, f"Cap\u00edtulo {m.group(1)}") for m in RE_CHAPTER.finditer(full)]
    found += [(m, "Pre\u00e2mbulo") for m in RE_PREAMBULO.finditer(full)]
    out: list[dict[str, Any]] = []
    for m, label in sorted(found, key=lambda x: x[0].start()):
        pos = m.end()
        title: list[str] = []
        lead_in = None
        while True:
            lm = RE_NEXT_LINE.match(full, pos)
            if not lm:
                break
            line = lm.group(1).strip()
            if not line:
                pos = lm.end()
                continue
            if RE_LEAD_IN.match(line):
                lead_in = line
                pos = lm.end()
                break
            is_title = (
                line == line.upper()
                and re.search(r"[A-Z\u00c0-\u00dd]", line)
                and not RE_ITEM_HEAD.match(line)
                and not RE_ART_HEAD.match(line)
                and len(title) < 3
            )
            if not is_title:
                break
            title.append(line)
            pos = lm.end()
        context = label + (f" \u2013 {' '.join(title)}" if title else "")
        if lead_in:
            context += f". {lead_in}"
        out.append({"start": m.start(), "end": pos, "label": label, "context": context})
    return out


def _anchor_positions(full: str) -> list[int]:
    """Signature dates and exposição headers: no article/item may run past one."""
    pos = [m.start() for m in RE_DATE_SIGNED.finditer(full)]
    pos += [m.start() for m in RE_EXPOSICAO.finditer(full)]
    return sorted(pos)


def _next_after(sorted_pos: list[int], at: int, default: int) -> int:
    i = bisect.bisect_right(sorted_pos, at)
    return sorted_pos[i] if i < len(sorted_pos) else default


def _context_at(headings: list[dict[str, Any]], pos: int) -> str | None:
    ctx = None
    for h in headings:
        if h["start"] >= pos:
            break
        ctx = h["context"]
    return ctx


def _extract_articles(full: str, stops: list[int], body_limit: int) -> list[tuple[int, int, str, str]]:
    """Articles of the normative body; "Art. N" lines after body_limit (the
    exposição de motivos) quote other norms and are not articles of this one."""
    matches = [m for m in RE_ART_HEAD.finditer(full) if m.start() < body_limit]
    out: list[tuple[int, int, str, str]] = []
    for i, m in enumerate(matches):
        s = m.start()
        e = matches[i + 1].start() if i + 1 < len(matches) else len(full)
        e = min(e, _next_after(stops, s, len(full)))
        label = f"Art. {m.group(2)}"
        text = _body(full[s:e])
        if text:
            out.append((s, e, label, text))
    return out


def _extract_items(
    full: str,
    headings: list[dict[str, Any]],
    article_starts: list[int],
    anchors: list[int],
) -> list[tuple[int, int, str, str, str]]:
    """Roman-numeral items of chapters that have no articles (CEM preâmbulo,
    princípios, direitos, disposições gerais)."""
    out: list[tuple[int, int, str, str, str]] = []
    heading_starts = [h["start"] for h in headings]
    for h in headings:
        seg_end = min(
            _next_after(heading_starts, h["start"], len(full)),
            _next_after(article_starts, h["start"], len(full)),
            _next_after(anchors, h["start"], len(full)),
        )
        heads = list(RE_ITEM_HEAD.finditer(full, h["end"], seg_end))
        for j, im in enumerate(heads):
            s = im.start()
            e = heads[j + 1].start() if j + 1 < len(heads) else seg_end
            text = _body(full[s:e])
            if text:
                out.append((s, e, f"{h['label']}, {im.group(1)}", h["context"], text))
    return out


def _split_long_article(full: str, s: int, e: int) -> list[tuple[int, int, str]]:
    """Split [s, e) into caput-prefixed parts at incisos/parágrafos.

    Returns [(start, end, text)]; a single element when no split applies.
    """
    whole = _body(full[s:e])
    if len(whole) <= LONG_ARTICLE_CHARS:
        return [(s, e, whole)]
    cuts = [m.start() for m in RE_SUBDIVISION.finditer(full, s, e)]
    if not cuts:
        return [(s, e, whole)]
    caput = _body(full[s:cuts[0]])
    bounds = cuts + [e]
    parts: list[tuple[int, int, str]] = []
    g_start = cuts[0]
    for k in range(1, len(bounds)):
        if bounds[k] - g_start < PART_CHARS and k < len(bounds) - 1:
            continue
        group = _body(full[g_start:bounds[k]])
        if group:
            parts.append((g_start, bounds[k], f"{caput}\n{group}"))
        g_start = bounds[k]
    return parts or [(s, e, whole)]


def _extract_signatories(tail: str) -> list[dict[str, str]]:
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    names: list[str] = []
    roles: list[str] = []
    for ln in lines:
        if RE_NAME_LINE.match(ln) and len(ln.split()) >= 2 and len(ln) <= 80:
            if not RE_SIG_ROLE_INLINE.search(ln):
                names.append(_squeeze(ln))
                continue
        rm = RE_SIG_ROLE_INLINE.search(ln)
        if rm and len(ln) <= 60:
            roles.append(_squeeze(rm.group(1)).lower())
    pairs: list[dict[str, str]] = []
    for i in range(min(len(names), len(roles))):
        pairs.append({"name": names[i], "role": roles[i]})
    if not pairs and names:
        pairs = [{"name": n, "role": ""} for n in names[:5]]
    seen: set[tuple[str, str]] = set()
    uniq: list[dict[str, str]] = []
    for s in pairs:
        key = (s["name"], s["role"])
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq


def _find_closure_anchor(full: str, search_from: int) -> int | None:
    candidates: list[int] = []
    m = RE_DATE_SIGNED.search(full, search_from)
    if m:
        candidates.append(m.start())
    em = RE_EXPOSICAO.search(full, search_from)
    if em:
        candidates.append(em.start())
    return min(candidates) if candidates else None


def _extract_date_signed(full: str) -> str | None:
    m = RE_DATE_SIGNED.search(full)
    if not m:
        return None
    d = int(m.group(1))
    mo = _MONTHS.get(m.group(2).lower())
    if not mo:
        return None
    return f"{int(m.group(3)):04d}-{mo:02d}-{d:02d}"


def _extract_modifications(full: str) -> list[str]:
    m = RE_MODIFICADA.search(full)
    if not m:
        return []
    return [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]


def _extract_revokes(full: str) -> list[str]:
    out: list[str] = []
    for m in RE_REVOGA_EXPLICIT.finditer(full):
        out.append(_squeeze(m.group(1)))
    for m in RE_REVOGA_LIST.finditer(full):
        out.append(_squeeze(m.group(1)))
    return sorted(set(out))


def _passage(
    doc_id: str,
    kind: str,
    index: int,
    text: str,
    char_start: int,
    char_end: int,
    spans: list[tuple[int, int, int]],
    citation: str,
    label: str | None = None,
    context: str | None = None,
    part: int | None = None,
) -> dict[str, Any]:
    p0, p1 = _span_pages(char_start, char_end, spans)
    pid = f"{doc_id}__{kind}_{index:03d}" + (f"_p{part}" if part else "")
    return {
        "passage_id": pid,
        "document_id": doc_id,
        "kind": kind,
        "label": label,
        "citation": citation,
        "context": context,
        "page_start": p0,
        "page_end": p1,
        "char_start": char_start,
        "char_end": char_end,
        "language": LANGUAGE,
        "domain": DOMAIN,
        "text": _with_context(context, text),
    }


def parse_cfm_pdf(path: Path, unicode_form: str | None = "NFC") -> dict[str, Any]:
    pages, pdf_meta, fmt, full = _prepare_document(path, unicode_form)
    spans = _page_char_spans(pages)

    number, year, header_raw = _find_act(full, path.name)
    document_id = _canonical_document_id(number, year)
    citation_base = f"Resolu\u00e7\u00e3o CFM n\u00ba {number}/{year}"

    ementa = _extract_ementa(full)
    publication = _extract_publication(full)
    modified_by = _extract_modifications(full)
    revokes = _extract_revokes(full)
    date_signed = _extract_date_signed(full)

    exposicao = RE_EXPOSICAO.search(full)
    body_limit = exposicao.start() if exposicao else len(full)
    headings = [h for h in _extract_headings(full) if h["start"] < body_limit]
    anchors = _anchor_positions(full)
    stops = sorted([h["start"] for h in headings] + anchors)
    articles = _extract_articles(full, stops, body_limit)
    items = _extract_items(full, headings, [a[0] for a in articles], anchors)
    if articles:
        body_end = articles[0][0]
    else:
        r = RE_RESOLVE.search(full)
        body_end = r.end() if r else len(full)
    considerandos = _extract_considerandos(full, body_end)

    signature_block: tuple[int, int, str] | None = None
    exposicao_block: tuple[int, int, str] | None = None
    signatories: list[dict[str, str]] = []

    units = [(a[0], a[1]) for a in articles] + [(it[0], it[1]) for it in items]
    if units:
        last_s, last_e = max(units)
        anchor = _find_closure_anchor(full, last_s + 1)
        sig_start = anchor if anchor is not None and anchor > last_s else last_e
    else:
        anchor = _find_closure_anchor(full, body_end)
        sig_start = anchor if anchor is not None else body_end

    em = RE_EXPOSICAO.search(full, sig_start)
    if em:
        sig_end = em.start()
        exp_text = _body(full[em.start():])
        if exp_text:
            exposicao_block = (em.start(), len(full), exp_text)
    else:
        sig_end = len(full)

    if sig_start < sig_end:
        sig_text = _body(full[sig_start:sig_end])
        if sig_text:
            signature_block = (sig_start, sig_end, sig_text)
            signatories = _extract_signatories(full[sig_start:sig_end])

    passages: list[dict[str, Any]] = []
    if ementa:
        passages.append(
            _passage(
                document_id,
                "ementa",
                0,
                ementa[2],
                ementa[0],
                ementa[1],
                spans,
                citation=f"{citation_base} \u2014 ementa",
                label="ementa",
            )
        )
    for i, (s, e, text) in enumerate(considerandos):
        passages.append(
            _passage(
                document_id,
                "considerando",
                i,
                text,
                s,
                e,
                spans,
                citation=f"{citation_base}, considerando {i + 1}",
                label=f"CONSIDERANDO {i + 1}",
            )
        )
    for i, (s, e, label, _text) in enumerate(articles):
        context = _context_at(headings, s)
        parts = _split_long_article(full, s, e)
        multi = len(parts) > 1
        for k, (ps, pe, text) in enumerate(parts, start=1):
            passages.append(
                _passage(
                    document_id,
                    "artigo",
                    i,
                    text,
                    ps,
                    pe,
                    spans,
                    citation=f"{citation_base}, {label}",
                    label=f"{label} (parte {k}/{len(parts)})" if multi else label,
                    context=context,
                    part=k if multi else None,
                )
            )
    for i, (s, e, label, context, text) in enumerate(items):
        passages.append(
            _passage(
                document_id,
                "item",
                i,
                text,
                s,
                e,
                spans,
                citation=f"{citation_base}, {label}",
                label=label,
                context=context,
            )
        )
    if signature_block:
        s, e, text = signature_block
        passages.append(
            _passage(
                document_id,
                "assinatura",
                0,
                text,
                s,
                e,
                spans,
                citation=f"{citation_base} \u2014 assinaturas / fecho",
                label="assinaturas",
            )
        )
    if exposicao_block:
        s, e, text = exposicao_block
        passages.append(
            _passage(
                document_id,
                "exposicao_motivos",
                0,
                text,
                s,
                e,
                spans,
                citation=f"{citation_base} \u2014 exposi\u00e7\u00e3o de motivos",
                label="exposi\u00e7\u00e3o de motivos",
            )
        )

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "issuer": "CFM",
        "act_type": "resolu\u00e7\u00e3o",
        "number": number,
        "year": year,
        "header_raw": header_raw or None,
        "ementa": ementa[2] if ementa else None,
        "publication": publication or None,
        "modified_by": modified_by,
        "revokes": revokes,
        "date_signed": date_signed,
        "signatories": signatories,
        "counts": {
            "considerandos": len(considerandos),
            "articles": len(articles),
            "items": len(items),
            "signature_block": 1 if signature_block else 0,
            "exposicao_motivos": 1 if exposicao_block else 0,
            "passages": len(passages),
        },
        "source": {
            "filename": path.name,
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "format": fmt,
            "page_count": pdf_meta.get("page_count") if isinstance(pdf_meta, dict) else None,
            "pdf_metadata": pdf_meta,
            "unicode_normalization": unicode_form,
            "language": LANGUAGE,
            "domain": DOMAIN,
        },
        "passages": passages,
    }
    return record


def _mark_duplicates(records: list[dict[str, Any]]) -> int:
    """Flag passages whose text (without context) repeats an earlier one."""
    first: dict[str, dict[str, Any]] = {}
    marked = 0
    for rec in records:
        for p in rec["passages"]:
            if p["kind"] not in DEDUPE_KINDS:
                continue
            body = p["text"]
            ctx = p.get("context")
            if ctx and body.startswith(ctx):
                body = body[len(ctx):]
            key = _squeeze(body).casefold()
            kept = first.get(key)
            if kept is None:
                first[key] = p
                continue
            p["duplicate_of"] = kept["passage_id"]
            kept.setdefault("also_cited_as", []).append(p["citation"])
            marked += 1
    return marked


def _iter_inputs(args_inputs: list[Path], input_dir: Path | None) -> Iterable[Path]:
    seen: set[Path] = set()
    for p in args_inputs:
        if p.is_dir():
            for q in sorted(p.glob("*.pdf")):
                if q not in seen:
                    seen.add(q)
                    yield q
        elif p.is_file():
            if p not in seen:
                seen.add(p)
                yield p
    if input_dir is not None:
        for q in sorted(input_dir.glob("*.pdf")):
            if q not in seen:
                seen.add(q)
                yield q


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest CFM legal PDFs into structured JSON + passages JSONL + manifest.",
    )
    ap.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="PDF files or directories to ingest.",
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of PDFs to ingest (e.g. legal_documents).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory (default: ./data).",
    )
    ap.add_argument(
        "--unicode-form",
        choices=("none", "NFC", "NFD", "NFKC", "NFKD"),
        default="NFC",
    )
    args = ap.parse_args()

    inputs = list(_iter_inputs(args.inputs, args.input_dir))
    if not inputs:
        print("Error: no input PDFs. Use --input-dir or pass files.", file=sys.stderr)
        sys.exit(1)

    unicode_form = None if args.unicode_form == "none" else args.unicode_form
    out_dir: Path = args.output_dir
    extracted_dir = out_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    passages_path = out_dir / "passages.jsonl"
    manifest_path = out_dir / "manifest.json"

    parsed: list[tuple[Path, dict[str, Any]]] = []
    for pdf_path in inputs:
        try:
            parsed.append((pdf_path, parse_cfm_pdf(pdf_path, unicode_form=unicode_form)))
        except Exception as e:
            print(f"[skip] {pdf_path.name}: {e}", file=sys.stderr)
    duplicates = _mark_duplicates([rec for _, rec in parsed])

    manifest_entries: list[dict[str, Any]] = []
    total_passages = 0
    with passages_path.open("w", encoding="utf-8") as pf:
        for pdf_path, rec in parsed:
            doc_id = rec["document_id"]
            extracted_path = extracted_dir / f"{doc_id}.json"
            extracted_path.write_text(
                json.dumps(rec, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            for p in rec["passages"]:
                pf.write(json.dumps(p, ensure_ascii=False))
                pf.write("\n")
                total_passages += 1
            manifest_entries.append(
                {
                    "document_id": doc_id,
                    "source_path": str(pdf_path.resolve()),
                    "extracted_path": str(extracted_path.resolve()),
                    "sha256": rec["source"]["sha256"],
                    "issuer": rec["issuer"],
                    "act_type": rec["act_type"],
                    "number": rec["number"],
                    "year": rec["year"],
                    "ementa": rec["ementa"],
                    "page_count": rec["source"].get("page_count"),
                    "counts": rec["counts"],
                }
            )
            print(
                f"[ok]  {pdf_path.name}  -> {doc_id}  "
                f"(arts={rec['counts']['articles']}, "
                f"items={rec['counts']['items']}, "
                f"considerandos={rec['counts']['considerandos']}, "
                f"passages={rec['counts']['passages']})"
            )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema": "cfm_manifest",
        "schema_version": SCHEMA_VERSION,
        "domain": DOMAIN,
        "language": LANGUAGE,
        "unicode_normalization": unicode_form,
        "document_count": len(manifest_entries),
        "passage_count": total_passages,
        "duplicate_passages": duplicates,
        "documents": manifest_entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"\nWrote: {manifest_path}  ({len(manifest_entries)} docs, "
        f"{total_passages} passages, {duplicates} marked duplicate_of)\n"
        f"       {passages_path}\n"
        f"       {extracted_dir}/"
    )


if __name__ == "__main__":
    main()
