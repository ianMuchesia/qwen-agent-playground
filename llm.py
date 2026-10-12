"""LLM backends.

OllamaLLM talks to a real local model (Qwen by default).
MockLLM fakes one with keyword rules so you can test the wiring with no model.

Both expose the same method: chat(messages, schema) -> str (raw JSON text).
Everything else in the playground only depends on that one method, which is
why you can swap models freely.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    """The model server couldn't be reached or returned an error."""


class OllamaLLM:
    """Calls Ollama's /api/chat endpoint.

    Passing a JSON schema as `format` turns on constrained decoding: the model
    is only allowed to emit tokens that keep the output valid against the
    schema. That is the single biggest reliability win for small models.
    """

    def __init__(self, model: str | None = None, host: str | None = None,
                 temperature: float = 0.0, timeout: float = 120.0):
        self.model = model or os.getenv("PLAYGROUND_MODEL", "qwen2.5:3b")
        host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if not host.startswith("http"):
            host = "http://" + host
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.calls = 0

    def chat(self, messages: list[dict], schema: dict | None = None) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if schema is not None:
            payload["format"] = schema
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise LLMError(f"Ollama returned HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise LLMError(
                f"Can't reach Ollama at {self.host} ({e.reason}). "
                f"Start it (open the Ollama app or run `ollama serve`) and pull the "
                f"model with `ollama pull {self.model}`."
            ) from e
        self.calls += 1
        return body["message"]["content"]


# ---------------------------------------------------------------------------
# Mock model: NOT smart. Keyword rules that always return schema-valid JSON,
# so you can run the CLI and the evals before Qwen is downloaded.
# ---------------------------------------------------------------------------

COMPLAINT_WORDS = ("worst", "terrible", "angry", "unacceptable", "broken",
                   "disappointed", "awful", "useless", "ridiculous")
SMALLTALK = ("hi", "hello", "hey", "thanks", "thank you", "good morning", "hi there")
QUESTION_STARTS = ("how", "what", "when", "where", "can", "do", "does", "is", "are", "which", "why")
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")
ORDER_RE = re.compile(r"\b\d{4}\b")


class MockLLM:
    def __init__(self):
        self.model = "mock"
        self.calls = 0

    def chat(self, messages: list[dict], schema: dict | None = None) -> str:
        self.calls += 1
        title = (schema or {}).get("title", "")
        # The first user message is the agent's real input (later ones are repair prompts).
        text = next(m["content"] for m in messages if m["role"] == "user")
        handler = getattr(self, f"_{title}", None)
        return json.dumps(handler(text) if handler else {})

    def _RouteDecision(self, text: str) -> dict:
        low = text.lower().strip(" !.?")
        if any(w in low for w in COMPLAINT_WORDS):
            return {"intent": "complaint", "confidence": 0.9, "reason": "negative words"}
        if ("password" in low and "reset" in low) or ("order" in low and ("where" in low or "track" in low)):
            return {"intent": "task", "confidence": 0.85, "reason": "asks for an action"}
        if low in SMALLTALK:
            return {"intent": "smalltalk", "confidence": 0.95, "reason": "greeting"}
        if text.strip().endswith("?") or low.split(" ")[0] in QUESTION_STARTS:
            return {"intent": "question", "confidence": 0.8, "reason": "looks like a question"}
        return {"intent": "unknown", "confidence": 0.3, "reason": "no rule matched"}

    def _Classification(self, text: str) -> dict:
        low = text.lower()
        if any(w in low for w in ("refund", "charge", "pay", "m-pesa", "invoice")):
            category = "billing"
        elif any(w in low for w in ("ship", "deliver", "order", "track", "package", "return")):
            category = "shipping"
        elif any(w in low for w in ("password", "login", "account", "email")):
            category = "account"
        else:
            category = "other"
        negative = any(w in low for w in COMPLAINT_WORDS)
        if any(w in low for w in ("urgent", "asap", "immediately")):
            priority = "urgent"
        elif negative:
            priority = "high"
        else:
            priority = "medium"
        sentiment = "negative" if negative else ("positive" if "thank" in low else "neutral")
        return {"category": category, "priority": priority, "sentiment": sentiment}

    def _ToolChoice(self, text: str) -> dict:
        low = text.lower()
        order = ORDER_RE.search(text)
        email = EMAIL_RE.search(text)
        if "order" in low and order:
            return {"tool": "lookup_order", "args": {"order_id": order.group()}, "reason": "order number given"}
        if "password" in low and email:
            return {"tool": "reset_password", "args": {"email": email.group()}, "reason": "email given"}
        return {"tool": "none", "args": {}, "reason": "missing details"}

    def _Answer(self, text: str) -> dict:
        m = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\[\d+\]|\nQUESTION:)", text, re.S)
        if not m:
            return {"answer": "I don't know.", "used_sources": [], "grounded": False}
        first = " ".join(re.split(r"(?<=[.!?])\s+", m.group(1).strip())[:2])
        return {"answer": first, "used_sources": [1], "grounded": True}

    def _Reply(self, text: str) -> dict:
        facts = text.split("FACTS:", 1)[-1].strip()
        if facts.startswith("none"):
            return {"reply": "Hi! How can I help you today?"}
        return {"reply": f"Here's an update: {facts}"}
