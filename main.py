import json
import os
import uuid

import psycopg
from fastapi import FastAPI, Header, HTTPException
from groq import Groq
from psycopg.types.json import Jsonb
from pydantic import BaseModel

app = FastAPI()
client = Groq()  # reads GROQ_API_KEY from the environment
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
APP_TOKEN = os.environ["APP_TOKEN"]
DATABASE_URL = os.environ.get("DATABASE_URL")  # Neon connection string

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
If the tool returns an error, tell the user the invoice was NOT saved and ask them to try again.
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

# Used only if DATABASE_URL is not set (invoices then vanish on restart).
INVOICES = {}
_table_ready = False


def db():
    # prepare_threshold=None keeps this safe with Neon's connection pooler.
    return psycopg.connect(DATABASE_URL, prepare_threshold=None, connect_timeout=10)


def ensure_table():
    global _table_ready
    if _table_ready or not DATABASE_URL:
        return
    with db() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS invoices ("
            "id TEXT PRIMARY KEY, "
            "data JSONB NOT NULL, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
    _table_ready = True


def save_invoice(invoice):
    if DATABASE_URL:
        ensure_table()
        with db() as conn:
            conn.execute(
                "INSERT INTO invoices (id, data) VALUES (%s, %s)",
                (invoice["id"], Jsonb(invoice)),
            )
    else:
        INVOICES[invoice["id"]] = invoice


def load_invoices():
    if DATABASE_URL:
        ensure_table()
        with db() as conn:
            rows = conn.execute(
                "SELECT data FROM invoices ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        return [r[0] for r in rows]
    return list(INVOICES.values())


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
                "id": "INV-" + uuid.uuid4().hex[:8].upper(),
                "customer": args["customer"],
                "currency": currency,
                "items": lines,
                "total": total,
                "workings": workings,
                "total_text": f"Total: {currency} {total:,.2f}",
            }
            save_invoice(invoice)
            return invoice
        return {"error": "unknown tool"}
    except (KeyError, TypeError, ValueError) as e:
        return {"error": f"bad tool input: {e}"}
    except psycopg.Error as e:
        print("Database error:", e)
        return {"error": "could not save the invoice, please try again"}


class Msg(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[Msg]


@app.get("/")
def health():
    return {"status": "ok"}


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
    try:
        return load_invoices()
    except psycopg.Error as e:
        print("Database error:", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
