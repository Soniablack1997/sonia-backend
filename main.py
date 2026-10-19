import json
import os
import uuid

from fastapi import FastAPI, Header, HTTPException
from groq import Groq
from pydantic import BaseModel

app = FastAPI()
client = Groq()  # reads GROQ_API_KEY from the environment
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
APP_TOKEN = os.environ["APP_TOKEN"]

SYSTEM = """You are a friendly, sharp business assistant for small business owners.
Keep replies short and practical. Reply in the language the user writes in
(including Pidgin, Yoruba, Hausa, or Igbo).
Before creating an invoice, confirm the customer name, items, quantities and prices.
Never invent prices or customer details. If something is missing, ask."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_invoice",
            "description": "Create an invoice for a customer once all details are confirmed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer": {"type": "string"},
                    "currency": {"type": "string", "description": "e.g. NGN, USD"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "quantity": {"type": "number"},
                                "unit_price": {"type": "number"},
                            },
                            "required": ["description", "quantity", "unit_price"],
                        },
                    },
                },
                "required": ["customer", "items"],
            },
        },
    }
]

INVOICES = {}


def run_tool(name, args):
    try:
        if name == "create_invoice":
            items = args["items"]
            total = sum(i["quantity"] * i["unit_price"] for i in items)
            invoice = {
                "id": "INV-" + uuid.uuid4().hex[:6].upper(),
                "customer": args["customer"],
                "currency": args.get("currency", "NGN"),
                "items": items,
                "total": total,
            }
            INVOICES[invoice["id"]] = invoice
            return invoice
        return {"error": "unknown tool"}
    except (KeyError, TypeError) as e:
        return {"error": f"bad tool input: {e}"}


class Msg(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[Msg]


@app.post("/chat")
def chat(body: ChatIn, x_app_token: str = Header(default="")):
    if x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    messages = [{"role": "system", "content": SYSTEM}]
    messages += [m.model_dump() for m in body.messages]

    for _ in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                max_tokens=1024,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Model error: {e}")

        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"reply": msg.content or ""}

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.function.name,
                            "arguments": c.function.arguments,
                        },
                    }
                    for c in msg.tool_calls
                ],
            }
        )
        for c in msg.tool_calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            output = run_tool(c.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": c.id,
                    "name": c.function.name,
                    "content": json.dumps(output),
                }
            )

    return {"reply": "Sorry, that took too many steps. Please try again."}


@app.get("/invoices")
def list_invoices(x_app_token: str = Header(default="")):
    if x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return list(INVOICES.values())
