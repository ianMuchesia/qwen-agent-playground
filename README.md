# Qwen multi-agent playground

A tiny support desk for a made-up shop ("Acme"), run by five agents that all share **one small model**.

```
message -> router ---> question  -> classifier -> search KB -> answerer  -> reply (+ source)
                  |--> task      -> classifier -> executor  -> tool      -> responder
                  |--> complaint -> classifier -> ticket    -> responder
                  |--> smalltalk -> responder
                  '--> unsure    -> escalate to a human (ticket)
```

The rule everywhere: **the model decides, the code enforces.** Every agent must return JSON that matches a schema; the code validates it, retries once with the error, and escalates if it still fails. Tools only run after their arguments pass validation, and sensitive tools wait for a human "yes".

## Setup (about 10 minutes)

1. Install **Python 3.10+** and **Ollama** (ollama.com).
2. Download the model (about 2 GB; runs on a laptop with 8 GB RAM, no GPU needed):
   ```
   ollama pull qwen2.5:3b
   ```
3. Install the one dependency:
   ```
   pip install -r requirements.txt
   ```
4. Test the wiring first with the fake model, then the real one:
   ```
   python main.py --mock
   python main.py --trace
   ```
   On some systems the command is `python3`, not `python`.

Try: `hi there`, `How long does standard shipping take?`, `Where is my order 1001?`, `Please reset my password, my email is sam@example.com`, `my package arrived broken, this is unacceptable`, `Do you sell gift vouchers?`

Inside the chat: `/trace` toggles the step log, `/tickets` shows tickets, `/quit` exits.

To try another model: `python main.py --model qwen3:4b` (or set `PLAYGROUND_MODEL`).

### Moving to another machine without downloading again (e.g. Linux at home)

The model is just files. Copy them on a USB stick instead of pulling 2 GB again.

1. **Copy the model:** take the whole `models` folder
   - Windows: `C:\Users\<you>\.ollama\models`
   - Linux (Ollama installed as a service): put it in `/usr/share/ollama/.ollama/models`, then
     ```
     sudo chown -R ollama:ollama /usr/share/ollama/.ollama/models
     sudo systemctl restart ollama
     ollama list    # should show qwen2.5:3b
     ```
   - Linux (you run `ollama serve` yourself): put it in `~/.ollama/models`.
2. **Ollama itself still needs installing** on Linux. On a fast connection, download the `ollama-linux-amd64` file from github.com/ollama/ollama/releases, carry it over, and unpack it with `sudo tar -C /usr -xf <file>`.
3. **The code:** copy this folder (or `git clone` it), **without** `.venv`. Then:
   ```
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

Or keep the model on one machine and point the code at it: `OLLAMA_HOST=http://<that-machine-ip>:11434` (the machine running Ollama must set `OLLAMA_HOST=0.0.0.0` to accept network connections).

## What each file is (and which stage it teaches)

| File | What it does | Stage |
|---|---|---|
| `llm.py` | Talks to Ollama with constrained JSON output; `MockLLM` fakes a model | 3 |
| `schemas.py` | The JSON contract for every agent | 3 |
| `agents.py` | `Agent` (prompt + schema + validate/retry) and `Trace` (step log) | 3, 7 |
| `orchestrator.py` | Routing, confidence threshold, escalation, all the prompts | 4 |
| `kb.py` + `data/kb/` | Keyword (BM25) search over markdown help articles | 5 |
| `context.py` | Token budget for what the answerer sees | 1 |
| `tools.py` | Tools with strict args and a human-approval gate | 2, 3 |
| `run_evals.py` + `evals/golden.jsonl` | Grades route + tool sequence; exits 1 on failure | 7 |

Read them bottom-up, so each file only uses things you've already seen: `schemas.py` -> `llm.py` -> `agents.py` -> `tools.py` -> `kb.py` -> `context.py` -> `orchestrator.py` -> `main.py` -> `run_evals.py`.

Notes from going through the code, and where to pick up next: [LEARNING_LOG.md](LEARNING_LOG.md).

## Your first run with the real model

Run `python run_evals.py`. **Expect some failures.** The mock passes 9/9 because it's rigged; a 3B model won't. That gap is the whole point: each failure is a lesson. Use `--verbose` to see the trace of every failing case and find the exact step that went wrong.

## Not in here yet
Conversation memory, durable workflows, security middleware, cost limits. Each can plug into this same code later.
