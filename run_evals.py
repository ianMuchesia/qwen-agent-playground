"""Stage 7 (starter): grade the agent's TRAJECTORY, not just its final words.

For every golden case we check:
  intent         - did the router pick the right route?
  tools          - did it call exactly the expected tools, in order?
  escalated      - did it hand off to a human when it should (and only then)?
  reply_contains - optional phrases the reply must include

Exits with code 1 if anything fails, so it can block a merge in CI.

    python run_evals.py            # real model
    python run_evals.py --mock     # fake model (checks the wiring only)
    python run_evals.py --verbose  # print traces of failing cases
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from kb import KnowledgeBase
from llm import LLMError, MockLLM, OllamaLLM
from orchestrator import Playground, deny_all
import tools

ROOT = Path(__file__).parent


def grade(case: dict, res) -> list[tuple[str, bool, object]]:
    checks = []
    if "intent" in case:
        checks.append(("intent", res.intent == case["intent"], res.intent))
    if "tools" in case:
        got = res.trace.tools_called()
        checks.append(("tools", got == case["tools"], got))
    if "escalated" in case:
        checks.append(("escalated", res.escalated == case["escalated"], res.escalated))
    for phrase in case.get("reply_contains", []):
        checks.append((f"says '{phrase}'", phrase.lower() in res.reply.lower(), res.reply[:80]))
    return checks


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true")
    p.add_argument("--model")
    p.add_argument("--file", default=str(ROOT / "evals" / "golden.jsonl"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    llm = MockLLM() if args.mock else OllamaLLM(model=args.model)
    # Approvals are always denied in evals: tests must never trigger real sensitive actions.
    pg = Playground(llm, KnowledgeBase(ROOT / "data" / "kb"), approve=deny_all)
    cases = [json.loads(line) for line in Path(args.file).read_text().splitlines() if line.strip()]

    print(f"Running {len(cases)} cases on {llm.model}\n")
    passed = 0
    total_calls = 0
    t0 = time.perf_counter()
    for case in cases:
        tools.TICKETS.clear()
        try:
            res = pg.handle(case["message"])
        except LLMError as e:
            print(f"  ! {e}")
            return 2
        checks = grade(case, res)
        ok = all(c[1] for c in checks)
        passed += ok
        total_calls += res.trace.llm_calls()
        print(f"  {'PASS' if ok else 'FAIL'}  {case['id']}")
        for name, good, got in checks:
            if not good:
                print(f"          x {name}: got {got!r}")
        if not ok and args.verbose:
            print(res.trace.pretty())

    secs = time.perf_counter() - t0
    print(f"\n{passed}/{len(cases)} passed  |  {total_calls} llm calls  |  {secs:.1f}s")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
