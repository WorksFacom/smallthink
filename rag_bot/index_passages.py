#!/usr/bin/env python3
"""Index CFM passages.jsonl into a Qdrant collection for the thinking bot.

Each point is one normative passage. The vector is built from
``citation + newline + text`` with the same Ollama embedding model used
at query time (default ``bge-m3``). Point ids are UUIDv5 of ``passage_id``,
so re-runs are idempotent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "cfm-passages")
DEFAULT_KINDS = "ementa,considerando,artigo,item"


def embed_batch(base_url: str, model: str, texts: list[str], timeout: int) -> list[list[float]]:
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    items = body.get("data") or []
    if len(items) != len(texts):
        raise RuntimeError(f"embedding count mismatch: got {len(items)}, expected {len(texts)}")
    return [it["embedding"] for it in items]


def load_passages(
    path: Path, kinds: set[str], limit: int | None, keep_duplicates: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if kinds and rec.get("kind") not in kinds:
                continue
            if rec.get("duplicate_of") and not keep_duplicates:
                continue
            if not rec.get("passage_id") or not rec.get("text") or not rec.get("citation"):
                continue
            rows.append(rec)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def vector_text(rec: dict[str, Any]) -> str:
    return f"{rec['citation']}\n{rec['text']}"


def point_id(passage_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, passage_id))


def existing_ids(client: QdrantClient, collection: str) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        try:
            points, offset = client.scroll(
                collection_name=collection,
                with_payload=False,
                with_vectors=False,
                limit=1024,
                offset=offset,
            )
        except Exception:
            return ids
        ids.update(str(p.id) for p in points)
        if offset is None:
            break
    return ids


def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool) -> None:
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
    for field in ("passage_id", "document_id", "kind", "citation"):
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def payload_of(rec: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "passage_id",
        "document_id",
        "kind",
        "label",
        "citation",
        "context",
        "also_cited_as",
        "page_start",
        "page_end",
        "text",
    )
    return {k: rec.get(k) for k in keep if rec.get(k) is not None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Index CFM passages into Qdrant.")
    ap.add_argument("--input", type=Path, default=Path("/data/passages.jsonl"))
    ap.add_argument("--qdrant-url", default="http://host.docker.internal:6333")
    ap.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT_API_KEY") or None)
    ap.add_argument("--collection", default="cfm_passages")
    ap.add_argument("--base-url", default="http://host.docker.internal:11434")
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--kinds", default=DEFAULT_KINDS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Also index passages marked duplicate_of (same text as another document's passage)",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"Error: passages file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    records = load_passages(args.input, kinds, args.limit, args.keep_duplicates)
    if not records:
        print("No passages to index.", file=sys.stderr)
        sys.exit(1)

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    try:
        probe = embed_batch(args.base_url, args.embed_model, [vector_text(records[0])], args.timeout)[0]
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(
            f"Error: embedding failed ({e}). Is Ollama up and is '{args.embed_model}' pulled?",
            file=sys.stderr,
        )
        sys.exit(2)

    dim = len(probe)
    print(
        f"passages={len(records)} model={args.embed_model} dim={dim} "
        f"collection={args.collection}",
        file=sys.stderr,
    )
    ensure_collection(client, args.collection, dim, args.recreate)

    skip: set[str] = set()
    if args.resume and not args.recreate:
        skip = existing_ids(client, args.collection)

    work = [rec for rec in records if point_id(rec["passage_id"]) not in skip]
    if not work:
        print("Nothing to do (all passages already indexed).")
        return

    # Reuse the probe vector for the first record when it is still in `work`.
    start = time.time()
    upserted = 0
    for i in range(0, len(work), args.batch_size):
        chunk = work[i : i + args.batch_size]
        texts = [vector_text(rec) for rec in chunk]
        if i == 0 and chunk[0]["passage_id"] == records[0]["passage_id"]:
            rest = embed_batch(args.base_url, args.embed_model, texts[1:], args.timeout) if len(texts) > 1 else []
            vecs = [probe, *rest]
        else:
            vecs = embed_batch(args.base_url, args.embed_model, texts, args.timeout)
        points = [
            qm.PointStruct(id=point_id(rec["passage_id"]), vector=vec, payload=payload_of(rec))
            for rec, vec in zip(chunk, vecs)
        ]
        client.upsert(collection_name=args.collection, points=points, wait=True)
        upserted += len(points)
        print(f"upserted {upserted}/{len(work)}", file=sys.stderr)

    print(f"Done. upserted={upserted} elapsed={time.time() - start:0.1f}s")


if __name__ == "__main__":
    main()
