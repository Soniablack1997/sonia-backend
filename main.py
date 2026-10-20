import json
import os
import uuid
from datetime import datetime, timedelta, timezone

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

# Lagos is UTC+1 all year (no daylight saving).
TZ = timezone(timedelta(hours=1))

SYSTEM_BASE = """You are Sonia, a friendly, sharp assistant for small business owners.
Keep replies short and practical. Reply in the language the user writes in
(including Pidgin, Yoruba, Hausa, or Igbo).
If the user writes in Nigerian Pidgin, reply in Nigerian Pidgin, not English.

What you can do: create invoices, do calculations, and prepare phone actions
(set an alarm, start a timer, add a calendar event). You cannot do anything else
yet, so never offer other things such as expense tracking, calls or messages.

Invoices: before creating one, confirm the customer name, items, quantities, prices
and currency. Never invent prices or customer details. If something is missing, ask.
When you create an invoice, show your workings using the figures returned by the tool:
the invoice number, the customer, one line per item written as
quantity x unit price = line total, then the total. Never do invoice arithmetic
yourself; copy the numbers from the tool result.
If the tool returns an error, tell the user the invoice was NOT saved.

Phone actions: they are NOT done until the user taps Confirm on their phone. After
calling one, say something short like "Tap Confirm to set it". Never say it is
already done. Use these tools only when the user asks. For a time of day use set_alarm,
for a countdown use set_timer, for a specific date and time use add_calendar_event.
Work out dates from the current date and time given below.

Reply in plain text only. Never use markdown, asterisks, pipes, or tables.
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
    },
    {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": "Prepare an alarm at a time of day on the user's phone. The user must tap Confirm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer", "description": "0-23, 24-hour clock"},
                    "minute": {"type": "integer", "description": "0-59"},
                    "label": {"type": "string"},
                },
                "required": ["hour", "minute"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Prepare a countdown timer on the user's phone. The user must tap Confirm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "integer", "description": "Length in seconds, 1 to 86400"},
                    "label": {"type": "string"},
                },
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_calendar_event",
            "description": "Prepare a calendar event or reminder for a specific date and time. The user must tap Confirm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {
                        "type": "string",
                        "description": "Local Lagos time as ISO 8601, e.g. 2026-10-21T09:00",
                    },
                    "duration_minutes": {"type": "integer", "description": "Default 60"},
                    "description": {"type": "string"},
                },
                "required": ["title", "start"],
            },
        },
    },
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


def clean_label(value):
    return str(value or "").strip()[:80]


def fmt_duration(total):
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    parts = []
    if hours:
        parts.append(f"{hours} h")
    if minutes:
        parts.append(f"{minutes} min")
    if seconds:
        parts.append(f"{seconds} s")
    return " ".join(parts)


def run_tool(name, args, actions):
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

        if name == "set_alarm":
            hour = int(args["hour"])
            minute = int(args.get("minute", 0))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                return {"error": "hour must be 0-23 and minute 0-59"}
            label = clean_label(args.get("label"))
            summary = f"Set alarm for {hour:02d}:{minute:02d}"
            if label:
                summary += f" - {label}"
            actions.append(
                {
                    "type": "set_alarm",
                    "hour": hour,
                    "minute": minute,
                    "label": label,
                    "summary": summary,
                }
            )
            return {"status": "waiting for the user to tap Confirm on their phone"}

        if name == "set_timer":
            seconds = int(args["seconds"])
            if not (1 <= seconds <= 86400):
                return {"error": "seconds must be between 1 and 86400"}
            label = clean_label(args.get("label"))
            summary = f"Start timer for {fmt_duration(seconds)}"
            if label:
                summary += f" - {label}"
            actions.append(
                {
                    "type": "set_timer",
                    "seconds": seconds,
                    "label": label,
                    "summary": summary,
                }
            )
            return {"status": "waiting for the user to tap Confirm on their phone"}

        if name == "add_calendar_event":
            title = clean_label(args["title"])
            if not title:
                return {"error": "title is required"}
            start = datetime.fromisoformat(str(args["start"]))
            if start.tzinfo is None:
                start = start.replace(tzinfo=TZ)
            if start < datetime.now(TZ) - timedelta(minutes=1):
                return {"error": "that time is in the past; check the date and year"}
            duration = int(args.get("duration_minutes") or 60)
            if not (5 <= duration <= 1440):
                return {"error": "duration_minutes must be between 5 and 1440"}
            end = start + timedelta(minutes=duration)
            actions.append(
                {
                    "type": "add_calendar_event",
                    "title": title,
                    "description": str(args.get("description") or "")[:300],
                    "start_ms": int(start.timestamp() * 1000),
                    "end_ms": int(end.timestamp() * 1000),
                    "summary": f"Add to calendar: {title} on {start:%a %d %b, %H:%M}",
                }
            )
            return {"status": "waiting for the user to tap Confirm on their phone"}

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

    now = datetime.now(TZ)
    system = SYSTEM_BASE + f"\n\nCurrent date and time in Lagos: {now:%A, %d %B %Y, %H:%M}."
    messages = [{"role": "system", "content": system}]
    messages += [m.model_dump() for m in body.messages]
    actions = []

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
            return {
                "reply": (msg.content or "").replace("**", ""),
                "actions": actions,
            }

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
            output = run_tool(c.function.name, args, actions)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": c.id,
                    "name": c.function.name,
                    "content": json.dumps(output),
                }
            )

    return {"reply": "Sorry, that took too many steps. Please try again.", "actions": []}


@app.get("/invoices")
def list_invoices(x_app_token: str = Header(default="")):
    if x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        return load_invoices()
    except psycopg.Error as e:
        print("Database error:", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
