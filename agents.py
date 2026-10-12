"""The Agent class and the Trace that records every step.

An "agent" here is just: one system prompt + one output schema + the shared
model. Five agents, one Qwen. The model decides; this code enforces.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class AgentFailure(RuntimeError):
    """An agent kept producing output that failed validation."""


@dataclass
class Trace:
    """Step-by-step log of one request. Evals grade this, not just the reply (Stage 7)."""

    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, **data) -> None:
        self.steps.append({"step": step, **data})

    def tools_called(self) -> list[str]:
        return [s["tool"] for s in self.steps if s["step"] == "tool"]

    def llm_calls(self) -> int:
        return sum(1 for s in self.steps if "attempt" in s)

    def to_json(self) -> str:
        return json.dumps(self.steps, indent=2, default=str)

    def pretty(self) -> str:
        lines = ["  --- trace ---"]
        for i, s in enumerate(self.steps, 1):
            data = {k: v for k, v in s.items() if k != "step"}
            lines.append(f"  {i:>2}. {s['step']:<11} {json.dumps(data, default=str)[:220]}")
        return "\n".join(lines)


class Agent(Generic[T]):
    def __init__(self, name: str, system_prompt: str, output_model: type[T], llm, max_retries: int = 1):
        self.name = name
        self.output_model = output_model
        self.llm = llm
        self.max_retries = max_retries
        self.schema = output_model.model_json_schema()
        # Small models do better when the schema is also spelled out in the prompt.
        self.system_prompt = (
            f"{system_prompt.strip()}\n\n"
            f"Respond with ONLY a JSON object matching this schema:\n{json.dumps(self.schema)}"
        )

    def run(self, user_content: str, trace: Trace) -> T:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        for attempt in range(1, self.max_retries + 2):
            t0 = time.perf_counter()
            raw = self.llm.chat(messages, schema=self.schema)
            ms = round((time.perf_counter() - t0) * 1000)
            try:
                out = self.output_model.model_validate_json(raw)
            except ValidationError as e:
                problems = "; ".join(
                    f"{'.'.join(map(str, err['loc'])) or 'root'}: {err['msg']}" for err in e.errors()[:3]
                )
                trace.add(self.name, ok=False, attempt=attempt, ms=ms, error=problems, raw=raw[:300])
                # Repair loop: show the model its mistake and ask again.
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That JSON was invalid ({problems}). "
                                                 "Reply again with only JSON that matches the schema."},
                ]
                continue
            trace.add(self.name, ok=True, attempt=attempt, ms=ms, output=out.model_dump())
            return out
        raise AgentFailure(f"{self.name} returned invalid output {self.max_retries + 1} times")
