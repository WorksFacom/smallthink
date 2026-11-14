#!/usr/bin/env python3
"""Terminal chat: question → CFM passages → fine-tuned model.

Type a question and press Enter. Commands:
  /sources   toggle showing retrieved passages (default on)
  /thought   toggle showing the <Thought> block (default off)
  /k N       set how many passages to retrieve (1–5)
  /quit      exit (Ctrl-D also works); unloads the model from Ollama
"""

from __future__ import annotations

import argparse
import sys

from app import rag

BANNER = (
    "Assistente CFM — responde só com base nos trechos normativos recuperados.\n"
    "Comandos: /sources  /thought  /k N  /quit\n"
)


def _print_sources(passages: list[dict]) -> None:
    if not passages:
        print("  (nenhum trecho acima do limiar de similaridade)")
        return
    for i, p in enumerate(passages, start=1):
        score = f" [{p['score']:.3f}]" if p.get("score") is not None else ""
        print(f"  [{i}] {p.get('citation')}{score}")
        text = " ".join((p.get("text") or "").split())
        if len(text) > 240:
            text = text[:239] + "…"
        print(f"      {text}")


def answer_once(question: str, *, k: int | None, show_sources: bool, show_thought: bool) -> None:
    qdrant = rag.client()
    try:
        passages = rag.retrieve(qdrant, question, k=k)
    except Exception as e:
        low = str(e).lower()
        if "not found" in low or "doesn't exist" in low or "404" in low:
            print(
                f"Coleção '{rag.COLLECTION}' não existe. Rode primeiro:\n"
                "  docker compose --profile index run --rm index",
                file=sys.stderr,
            )
        else:
            print(f"Erro na recuperação: {e}", file=sys.stderr)
        return

    if show_sources:
        print("\nTrechos:")
        _print_sources(passages)

    if not passages and rag.ABSTAIN_WITHOUT_SOURCES:
        print(f"\nAssistente:\n{rag.NO_SOURCES_ANSWER}\n")
        return

    prompt = rag.build_prompt(question, passages)
    print("\nAssistente:", flush=True)
    parts: list[str] = []
    try:
        for piece in rag.generate_stream(prompt):
            parts.append(piece)
            if show_thought:
                sys.stdout.write(piece)
                sys.stdout.flush()
    except RuntimeError as e:
        print(f"\nErro na geração: {e}", file=sys.stderr)
        return

    raw = "".join(parts)
    answer, thought = rag.parse_answer(raw)
    if show_thought:
        # Streaming already printed the raw text; add a clean final answer.
        print("\n\nResposta:")
    print(answer or raw)
    if not show_thought and thought:
        print("\n(/thought para ver o raciocínio)")
    print()


def repl(k: int | None, show_sources: bool, show_thought: bool) -> None:
    print(BANNER)
    try:
        while True:
            try:
                line = input("Você> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not line:
                continue
            if line in ("/quit", "/exit", "/q"):
                return
            if line == "/sources":
                show_sources = not show_sources
                print(f"mostrar trechos: {'on' if show_sources else 'off'}")
                continue
            if line == "/thought":
                show_thought = not show_thought
                print(f"mostrar raciocínio: {'on' if show_thought else 'off'}")
                continue
            if line.startswith("/k"):
                try:
                    n = int(line.split(maxsplit=1)[1])
                    if not 1 <= n <= 5:
                        raise ValueError
                    k = n
                    print(f"top_k = {k}")
                except (IndexError, ValueError):
                    print("uso: /k N  (1–5)")
                continue
            if line.startswith("/"):
                print("comando desconhecido")
                continue
            answer_once(line, k=k, show_sources=show_sources, show_thought=show_thought)
    finally:
        rag.unload()


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat no terminal com o assistente CFM.")
    ap.add_argument("question", nargs="*", help="Pergunta única (sem REPL).")
    ap.add_argument("--k", type=int, default=None, help="Trechos a recuperar (1–5).")
    ap.add_argument("--no-sources", action="store_true", help="Não listar trechos.")
    ap.add_argument("--thought", action="store_true", help="Mostrar o bloco <Thought>.")
    args = ap.parse_args()

    if args.question:
        try:
            answer_once(
                " ".join(args.question),
                k=args.k,
                show_sources=not args.no_sources,
                show_thought=args.thought,
            )
        finally:
            rag.unload()
        return
    repl(args.k, not args.no_sources, args.thought)


if __name__ == "__main__":
    main()
