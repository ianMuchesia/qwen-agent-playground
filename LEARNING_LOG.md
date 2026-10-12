# Learning log

Notes from walking through the code, file by file, in simple words.
Reading order: schemas -> llm -> agents -> tools -> kb -> context -> orchestrator -> main -> run_evals.

## Session 1 (2026-10-12)

**Setup done:** Ollama 0.40.3 installed, `qwen2.5:3b` pulled, `.venv` created, `run_evals.py --mock` passes 9/9.
Real-model evals not run yet.

### The big idea
One small model, used five times with five different "hats" (system prompts): router, classifier,
answerer, executor, responder. **The model decides, the code enforces.** The model only fills in a
JSON form; Python checks it and does the real work.

### 1. schemas.py: the contracts
- One class per agent = the exact JSON that agent must return.
- **Who reads the field decides its type:**
  - the *code* reads it (`intent`, `tool`, `priority`) -> `Literal[...]`, because `if` statements need exact values
  - a *human* reads it (`reply`, `reason`, `answer`) -> plain `str`
- A `Literal` is a promise: every value in it must be handled by code. Adding `"refund"` to `Intent` means also changing:
  the handler dict in `orchestrator.py` (or it crashes with `KeyError`), `ROUTER_PROMPT`, a new `_refund` function,
  `MockLLM`, and `golden.jsonl`. A new *tool* is only needed if refunds become an action.
- Give the model a safe way out: `grounded: false`, `tool: "none"`. Otherwise small models make things up.
- Design the output first, then the prompt.

### 2. llm.py: real model vs fake model
- `OllamaLLM` = the real Qwen. It just carries messages back and forth.
- `MockLLM` = a fake that pretends using word rules. The keyword/category code lives **only** here.
- Both have the same method: `chat(messages, schema) -> JSON text`. The rest of the code only calls `chat()`
  and doesn't know which one it's using (like a charger plug). `--mock` swaps them in `main.py`.
- With the real model, **Qwen** makes the route decision; the code checks it and follows it.
- Router = decides the **path**. Classifier = adds **labels** (they don't change prompts yet; idea for later).

### 3. agents.py: Agent and Trace
- `Agent.run()` is the **only** place that calls `llm.chat()`. All five agents are `Agent`s with different prompts.
- `Trace` = a diary for **one message**. Agents *and* the orchestrator write to it (`input`, `retrieve`, `tool`, `escalate`).
  Used for: `--trace` output, evals grading, counting LLM calls (failed tries count too, and each costs time/money).
- **Repair loop:** bad JSON -> the error goes back to **Qwen** (not the customer) -> try again -> still bad ->
  `AgentFailure` -> orchestrator escalates to a human ticket. 2 tries total (`max_retries=1`).
- Ollama `format` makes the JSON valid; Pydantic checks the rules Ollama doesn't (e.g. `confidence` <= 1).
- **Network errors are NOT retried.** `LLMError` goes straight to `main.py`. (Idea for later: retry with a wait.)

### 4. tools.py (started)
- The executor only **proposes** a tool. `run_tool` has 3 checks before anything runs:
  1. Does the tool exist? 2. Are the args valid? 3. If it needs approval, did **staff** say yes?
- Approval comes from staff, not the customer (a scammer can type "yes" too). Anything auth/account = sensitive.
- Only add approval where a mistake hurts (`reset_password` yes, `lookup_order` no). Too many approvals and people click yes without reading.
- Evals use `deny_all`: nobody is watching, and tests must never trigger real actions.

## Next session
- Finish tools.py (`describe_tools()` feeding the executor prompt), then kb.py + `data/kb/`.
- Run `python run_evals.py --verbose` on real Qwen and go through each failure.
