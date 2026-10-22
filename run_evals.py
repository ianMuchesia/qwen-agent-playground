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
    python run_evals.py --show-all # print traces of every case, passing too

Every run is also saved to evals/runs/<time>_<model>.jsonl: one line per case with
the full trace (every agent's output, untruncated), so you can compare runs and models.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
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
    p.add_argument("--verbose", action="store_true", help="print traces of failing cases")
    p.add_argument("--show-all", action="store_true", help="print traces of every case")
    args = p.parse_args()

    llm = MockLLM() if args.mock else OllamaLLM(model=args.model)
    # Approvals are always denied in evals: tests must never trigger real sensitive actions.
    pg = Playground(llm, KnowledgeBase(ROOT / "data" / "kb"), approve=deny_all)
    cases = [json.loads(line) for line in Path(args.file).read_text().splitlines() if line.strip()]

    runs_dir = ROOT / "evals" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / f"{datetime.now():%Y-%m-%d_%H%M%S}_{llm.model.replace(':', '-')}.jsonl"
    log = log_path.open("w", encoding="utf-8")

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
        if args.show_all or (not ok and args.verbose):
            print(res.trace.pretty())
        log.write(json.dumps({
            "id": case["id"], "passed": ok, "message": case["message"],
            "intent": res.intent, "escalated": res.escalated, "reply": res.reply,
            "llm_calls": res.trace.llm_calls(),
            "failed_checks": [{"check": n, "got": g} for n, good, g in checks if not good],
            "trace": res.trace.steps,
        }, default=str) + "\n")

    log.close()
    secs = time.perf_counter() - t0
    print(f"\n{passed}/{len(cases)} passed  |  {total_calls} llm calls  |  {secs:.1f}s")
    print(f"full log: {log_path.relative_to(ROOT)}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
