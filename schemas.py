"""Output contracts for every agent (Stage 3: strict schemas).

Each agent must answer with JSON matching one of these models. Literal[...]
fields force the model to pick from a fixed list instead of free-texting a
category, which is where small models are most reliable.
"""
from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["question", "task", "complaint", "smalltalk", "unknown"]


class RouteDecision(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1, description="0 = guessing, 1 = certain")
    reason: str = Field(max_length=200)
    # The router reads the long message once and hands forward the clean request,
    # so search and the answerer never have to deal with the backstory.
    search_query: str = Field(max_length=120, description="the customer's actual question or request in a few words")


class Classification(BaseModel):
    category: Literal["billing", "shipping", "account", "product", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    sentiment: Literal["positive", "neutral", "negative"]


class Answer(BaseModel):
    answer: str
    used_sources: list[int] = Field(description="numbers of the SOURCES you used")
    grounded: bool = Field(description="false if the sources don't contain the answer")


class ToolChoice(BaseModel):
    tool: Literal["lookup_order", "create_ticket", "reset_password", "none"]
    args: dict[str, str]
    reason: str = Field(max_length=200)


class Reply(BaseModel):
    reply: str
