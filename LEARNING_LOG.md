# Learning log

Notes from walking through the code, file by file, in simple words.
Reading order: schemas -> llm -> agents -> tools -> kb -> context -> orchestrator -> main -> run_evals.

## Session 1 (2026-10-12)

**Setup done:** Ollama 0.40.3 installed, `qwen2.5:3b` pulled, `.venv` created, `run_evals.py --mock` passes every case.
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

### 4. tools.py
- The executor only **proposes** a tool. `run_tool` has 3 checks before anything runs:
  1. Does the tool exist? 2. Are the args valid? 3. If it needs approval, did **staff** say yes?
- Approval comes from staff, not the customer (a scammer can type "yes" too). Anything auth/account = sensitive.
- Only add approval where a mistake hurts (`reset_password` yes, `lookup_order` no). Too many approvals and people click yes without reading.
- Evals use `deny_all`: nobody is watching, and tests must never trigger real actions.
- `describe_tools()` turns the `TOOLS` list into text for the executor prompt. Add a tool -> Qwen sees it automatically.
- Qwen only knows what the descriptions say: `priority (string)` never tells it the allowed values. Write good `description=`.
- Every tool costs prompt space on every call.
- Executor = turns a request into an action. It only **chooses** (tool + args, or `"none"`); `run_tool` checks and runs it.

## Session 2 (2026-10-14)

### Who decides what
- **Qwen** decides: route, labels, which tool, the wording.
- **Code** decides: is the route trusted (confidence >= 0.6), attach sources or not (only on the question route),
  run the tool or not (3 checks), escalate or not.
- Qwen picks the route **once**; after that the code follows a fixed plan for that route.
- One message = several Qwen calls: smalltalk 2, question 3, task 4, complaint 3, unsure 1 (+1 per retry).
- An agent = **prompt** (how to think) + **schema** (what it's allowed to answer) + **code** (what happens next).

### 5. kb.py: the R in RAG
- Cuts the help articles at `##` headings (12 chunks), scores each against the question with BM25 (word matching),
  returns the top 3. Rare words count more. Scores below 2.0 -> human.
- Weakness: matches words, not meaning ("shipping" vs "delivery"). Fix later: embeddings + BM25 = hybrid search.
  FAISS isn't needed for 12 chunks.

### 6. context.py: make it fit
- Only applies right before the **answerer**. Total budget 600 tokens, question max 150 (~100 words),
  each source max 250. 1 token ~ 4 characters (a guess).
- Fills in priority order: question first, then best source, weakest source cut first.
- Keeps the **start** of a text and cuts the **end**.

### The long-message bug (found with real Qwen) and the fix
- A long story with the question at the end: router + classifier understood it, but **search** used every word
  (found the wrong articles) and **context.py** cut off the real question. The safety nets worked (no made-up answer,
  sent to a human), but the customer got no help.
- Root cause: the router understood the question but only passed forward one word ("question").
  **Lesson: each agent should pass forward the clean result of its work, not just a label.**
- Fix: router now also returns `search_query` ("standard delivery time"). Search and the answerer use it;
  the full message is kept only for the human ticket. Now it finds "Delivery times" and answers "3 to 5 business days".
- Added a limit at the door: messages over 2,000 characters are refused with 0 model calls.
- Added eval `faq-long-story` so it can't come back.
- context.py is now a safety net (a seatbelt): it rarely needs to cut.

### Why limit length
- Small models get worse with long input. Ollama also has a window (`num_ctx`) and silently cuts the **start**
  (our instructions) if the prompt is too big.
- Paid models charge per token, and the message is resent to every agent.
- Slower: 15-30 s per Qwen call on this PC's CPU.
- Limit smartly: compress once early (one agent summarizes), cut only as a last resort.

## Session 3 (2026-10-15 and 2026-10-22): first real-model evals

### First run on qwen2.5:3b: 7/10 (32 calls, 238 s)
| Test | What went wrong |
|---|---|
| faq-returns | router called a question a **complaint** |
| abstain-unknown-topic | router called it a **complaint** (0.9 confidence!), then the responder **made up** a voucher policy |
| track-missing-number | executor **made up** order number 1234 (the format check passed: it's 4 digits) |

### Lessons
- **Look for the pattern, not single failures.** 2 of 3 were the router over-using "complaint". One prompt fix covers both.
- **Small models make things up** (hallucinate). They write what *sounds* likely: "1234" sounds like an order number.
  Defences: give the facts, give a way out (`none`, `grounded: false`), **check in code**, fixed sentences where
  the answer never changes, evals.
- **Prompt rules are not enough.** The prompt said "never make up order numbers"; Qwen did anyway.
- **Self-reported confidence is unreliable.** Qwen said 0.9 while wrong, so the 0.6 threshold couldn't help.
- **Right answer, wrong reason = still a failure.** On the complaint path Qwen guessed the return policy without
  seeing the article. The guess was right but missed "except for damaged items". The evals check the *path*, not just the reply.
- A format check (4 digits) can't tell a real value from an invented one.

### Fixes -> 10/10
1. Router prompt: `question` includes "can I...?" and "do you sell / offer / have X?";
   `complaint` ONLY when something already went wrong (late, broken, wrong charge).
2. Executor guard in `_execute`: if a tool arg isn't in the customer's message, treat it as `none` and ask for it.

### Logging
- Every eval run is saved to `evals/runs/<time>_<model>.jsonl`: full trace for every case, passed or failed.
- `run_evals.py --show-all` prints every trace. Use it to spot "barely passing" cases (confidence near 0.6)
  and slow agents (`ms`), and to compare models case by case.

## Ideas for later
- Merge router + classifier into one agent (reads the message once, saves ~15 s). Let the evals decide.
- Compare models: `run_evals.py --model qwen2.5:7b`, then compare pass rate, calls and time.
- A different model per agent: small for router/classifier, bigger for the answerer, frontier for escalation.
- Fixed sentence for complaint replies instead of the responder (can't make things up, one fewer call).
- Hybrid search (BM25 + embeddings with `nomic-embed-text`).
