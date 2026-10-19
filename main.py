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
If the user writes in Nigerian Pidgin, reply in Nigerian Pidgin, not English.
Before creating an invoice, confirm the customer name, items, quantities, prices and currency.
Never invent prices or customer details. If something is missing, ask.
Reply in plain text only. Never use markdown, asterisks, pipes, or tables.
When you create an invoice, show your workings in plain text using the figures
returned by the tool: the invoice number, the customer, one line per item written
as quantity x unit price = line total, then the total. Never do the arithmetic
yourself for an invoice; copy the numbers from the tool result.
For any other calculation, show the steps one per line."""

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
            currency = args.get("currency", "NGN")
            lines = []
            workings = []
            total = 0.0
            for item in args["items"]:
                qty = float(item["quantity"])
                price = float(item["unit_price"])
                line_total = qty * price
                total += line_total
                lines.append(
                    {
                        "description": item["description"],
                        "quantity": qty,
                        "unit_price": price,
                        "line_total": line_total,
                    }
                )
                workings.append(
                    f'{item["description"]}: {qty:g} x {price:,.2f} = {line_total:,.2f}'
                )
            invoice = {
                "id": "INV-" + uuid.uuid4().hex[:6].upper(),
                "customer": args["customer"],
                "currency": currency,
                "items": lines,
                "total": total,
                "workings": workings,
                "total_text": f"Total: {currency} {total:,.2f}",
            }
            INVOICES[invoice["id"]] = invoice
            return invoice
        return {"error": "unknown tool"}
    except (KeyError, TypeError, ValueError) as e:
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
            return {"reply": (msg.content or "").replace("**", "")}

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
    
