"""Chat with the playground.

    python main.py                 # real model via Ollama (qwen2.5:3b by default)
    python main.py --mock          # fake model, to test the wiring
    python main.py --model qwen3:4b --trace
    python main.py --once "Where is my order 1001?"

Commands inside the chat: /trace (toggle step log), /tickets, /quit
"""
from __future__ import annotations

import argparse
from pathlib import Path

from kb import KnowledgeBase
from llm import LLMError, MockLLM, OllamaLLM
from orchestrator import Playground
from tools import TICKETS

KB_DIR = Path(__file__).parent / "data" / "kb"


def cli_approve(tool: str, args: dict) -> bool:
    try:
        answer = input(f"\n  [approval needed] run {tool}({args})? [y/N] ")
    except EOFError:  # no human at the keyboard -> deny
        return False
    return answer.strip().lower() in ("y", "yes")


def show(res, show_trace: bool) -> None:
    meta = [res.intent or "?"]
    if res.classification:
        c = res.classification
        meta.append(f"{c['category']}/{c['priority']}/{c['sentiment']}")
    if res.trace.tools_called():
        meta.append("tools: " + ", ".join(res.trace.tools_called()))
    meta.append(f"{res.trace.llm_calls()} llm calls")
    if res.escalated:
        meta.append("ESCALATED")
    print(f"\nbot > {res.reply}")
    print(f"      [{' | '.join(meta)}]")
    if show_trace:
        print(res.trace.pretty())


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen multi-agent playground")
    p.add_argument("--mock", action="store_true", help="use the fake keyword model")
    p.add_argument("--model", help="Ollama model name (default qwen2.5:3b or $PLAYGROUND_MODEL)")
    p.add_argument("--trace", action="store_true", help="print every step")
    p.add_argument("--once", help="handle one message and exit")
    args = p.parse_args()

    llm = MockLLM() if args.mock else OllamaLLM(model=args.model)
    pg = Playground(llm, KnowledgeBase(KB_DIR), approve=cli_approve)

    if args.once:
        try:
            show(pg.handle(args.once), args.trace)
        except LLMError as e:
            print(f"\n  ! {e}")
        return

    print(f"Acme support playground  (model: {llm.model}, {len(pg.kb.chunks)} KB chunks)")
    print("Type a message. Commands: /trace  /tickets  /quit")
    show_trace = args.trace
    while True:
        try:
            msg = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg in ("/quit", "/exit"):
            break
        if msg == "/trace":
            show_trace = not show_trace
            print(f"  trace {'on' if show_trace else 'off'}")
            continue
        if msg == "/tickets":
            print("\n".join(f"  {t}" for t in TICKETS) or "  no tickets yet")
            continue
        try:
            show(pg.handle(msg), show_trace)
        except LLMError as e:
            print(f"\n  ! {e}")


if __name__ == "__main__":
    main()
