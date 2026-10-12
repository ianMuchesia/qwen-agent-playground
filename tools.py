"""Stage 3: tools the executor agent may use, each with a strict args schema.

The model only *proposes* a tool call. This code validates the arguments,
checks whether a human must approve (a tiny HITL gate, Stage 2), and only
then runs it. The "database" is in-memory and fake.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError

ORDERS = {
    "1001": {"status": "shipped", "carrier": "DHL", "eta": "2-3 business days"},
    "1002": {"status": "processing", "eta": "ships within 24 hours"},
    "1003": {"status": "delivered", "delivered_on": "2026-10-08"},
}
TICKETS: list[dict] = []


class LookupOrderArgs(BaseModel):
    order_id: str = Field(pattern=r"^\d{4}$", description="the customer's 4-digit order number")


class CreateTicketArgs(BaseModel):
    summary: str = Field(min_length=5, max_length=200)
    priority: Literal["low", "medium", "high", "urgent"] = "medium"


class ResetPasswordArgs(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$", description="the account email")


def lookup_order(a: LookupOrderArgs) -> dict:
    order = ORDERS.get(a.order_id)
    return {"found": False, "order_id": a.order_id} if order is None else {"found": True, "order_id": a.order_id, **order}


def create_ticket(a: CreateTicketArgs) -> dict:
    ticket = {"ticket_id": f"T-{len(TICKETS) + 1:04d}", "summary": a.summary, "priority": a.priority}
    TICKETS.append(ticket)
    return ticket


def reset_password(a: ResetPasswordArgs) -> dict:
    return {"sent": True, "email": a.email, "note": "reset link sent (simulated), expires in 1 hour"}


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], dict]
    needs_approval: bool = False  # sensitive actions wait for a human


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("lookup_order", "Look up an order's status by its order number.", LookupOrderArgs, lookup_order),
        Tool("create_ticket", "Open a support ticket for a human to follow up.", CreateTicketArgs, create_ticket),
        Tool("reset_password", "Email a password reset link to the account email.", ResetPasswordArgs,
             reset_password, needs_approval=True),
    ]
}


def describe_tools() -> str:
    """Tool list for the executor's prompt. These lines cost tokens on every call (Stage 1!)."""
    lines = []
    for t in TOOLS.values():
        props = t.args_model.model_json_schema().get("properties", {})
        args = ", ".join(f"{k} ({v.get('description', v.get('type', 'string'))})" for k, v in props.items())
        lines.append(f"- {t.name}: {t.description} Args: {args}")
    return "\n".join(lines)


def run_tool(name: str, args: dict, approve: Callable[[str, dict], bool]) -> dict:
    tool = TOOLS.get(name)
    if tool is None:
        return {"ok": False, "error": f"unknown tool '{name}'"}
    try:
        parsed = tool.args_model.model_validate(args)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:3])
        return {"ok": False, "error": f"invalid args: {problems}", "invalid_args": True}
    if tool.needs_approval and not approve(name, parsed.model_dump()):
        return {"ok": False, "error": "a staff member did not approve this action", "denied": True}
    return {"ok": True, "result": tool.fn(parsed)}


if __name__ == "__main__":
    print(describe_tools())
    print(json.dumps(run_tool("lookup_order", {"order_id": "1001"}, lambda *_: True), indent=2))
