import json
import os
import re
import uuid
from datetime import datetime

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

SYSTEM = """You are Sonia, a friendly, sharp assistant for small business owners.
Keep replies short and practical. Reply in the language the user writes in
(including Pidgin, Yoruba, Hausa, or Igbo).
If the user writes in Nigerian Pidgin, reply in Nigerian Pidgin, not English.

What you can do right now:
1. Create and save invoices.
2. Prepare phone actions: set an alarm, start a timer, add a calendar event,
   open the dialer with a number, or draft a text message. For these you only
   PREPARE the action. The user must tap Confirm on their phone, and for calls and
   texts they still press call or send themselves. Never say a phone action is done;
   say you have prepared it and ask them to tap Confirm.
You cannot do anything else yet. Do not offer other features such as expense
tracking, sending messages by yourself, or payments. If asked, say it is not
available yet.

Before creating an invoice, confirm the customer name, items, quantities, prices and currency.
Never invent prices, phone numbers or customer details. If something is missing, ask.
Reply in plain text only. Never use markdown, asterisks, pipes, or tables.
When you create an invoice, show your workings in plain text using the figures
returned by the tool: the invoice number, the customer, one line per item written
as quantity x unit price = line total, then the total. Never do the arithmetic
yourself for an invoice; copy the numbers from the tool result.
If a tool returns an error, tell the user it did not work and why, in simple words.
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
            "description": "Prepare an alarm on the user's phone at a clock time (24-hour).",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer", "description": "0-23"},
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
            "description": "Prepare a countdown timer on the user's phone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "integer", "description": "1 to 86400"},
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
            "description": "Prepare a calendar event on the user's phone. Times are the user's local time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string", "description": "YYYY-MM-DDTHH:MM, e.g. 2026-10-21T09:00"},
                    "end": {"type": "string", "description": "YYYY-MM-DDTHH:MM, optional"},
                    "location": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["title", "start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dial_number",
            "description": "Prepare a phone call by opening the dialer with the number filled in.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string"},
                    "label": {"type": "string", "description": "who is being called"},
                },
                "required": ["phone_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_sms",
            "description": "Prepare a text message by opening the messaging app with the text filled in.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["phone_number", "message"],
            },
        },
    },
]

DEVICE_TOOLS = {"set_alarm", "set_timer", "add_calendar_event", "dial_number", "draft_sms"}
PHONE_RE = re.compile(r"^\+?[0-9 ()\-]{5,20}$")
TIME_FMT = "%Y-%m-%dT%H:%M"

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


def prepare_device_action(name, args):
    """Validate a phone action. Returns (action, error). The app runs it after the user confirms."""
    try:
        if name == "set_alarm":
            hour, minute = int(args["hour"]), int(args["minute"])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                return None, {"error": "hour must be 0-23 and minute 0-59"}
            label = str(args.get("label", "")).strip()
            summary = f"Set alarm for {hour:02d}:{minute:02d}" + (f" ({label})" if label else "")
            return {"name": name, "summary": summary,
                    "args": {"hour": hour, "minute": minute, "label": label}}, None

        if name == "set_timer":
            seconds = int(args["seconds"])
            if not (1 <= seconds <= 86400):
                return None, {"error": "seconds must be between 1 and 86400"}
            label = str(args.get("label", "")).strip()
            mins, secs = divmod(seconds, 60)
            hrs, mins = divmod(mins, 60)
            parts = [f"{hrs}h" if hrs else "", f"{mins}m" if mins else "", f"{secs}s" if secs else ""]
            summary = "Start timer for " + " ".join(p for p in parts if p) + (f" ({label})" if label else "")
            return {"name": name, "summary": summary,
                    "args": {"seconds": seconds, "label": label}}, None

        if name == "add_calendar_event":
            title = str(args["title"]).strip()
            if not title:
                return None, {"error": "title is empty"}
            start = datetime.strptime(str(args["start"]), TIME_FMT)
            end_raw = str(args.get("end", "")).strip()
            end = datetime.strptime(end_raw, TIME_FMT) if end_raw else None
            if end and end <= start:
                return None, {"error": "end must be after start"}
            when = start.strftime("%a %d %b %Y at %H:%M")
            summary = f"Add to calendar: {title}, {when}"
            return {"name": name, "summary": summary,
                    "args": {"title": title,
                             "start": start.strftime(TIME_FMT),
                             "end": end.strftime(TIME_FMT) if end else "",
                             "location": str(args.get("location", "")).strip(),
                             "notes": str(args.get("notes", "")).strip()}}, None

        if name == "dial_number":
            phone = str(args["phone_number"]).strip()
            if not PHONE_RE.match(phone):
                return None, {"error": "that does not look like a phone number"}
            label = str(args.get("label", "")).strip()
            summary = f"Open dialer: {phone}" + (f" ({label})" if label else "")
            return {"name": name, "summary": summary,
                    "args": {"phone_number": phone, "label": label}}, None

        if name == "draft_sms":
            phone = str(args["phone_number"]).strip()
            message = str(args["message"]).strip()
            if not PHONE_RE.match(phone):
                return None, {"error": "that does not look like a phone number"}
            if not message:
                return None, {"error": "message is empty"}
            summary = f'Draft text to {phone}: "{message}"'
            return {"name": name, "summary": summary,
                    "args": {"phone_number": phone, "message": message}}, None
    except (KeyError, TypeError, ValueError) as e:
        return None, {"error": f"bad input: {e}"}
    return None, {"error": "unknown tool"}


class Msg(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[Msg]
    now: str = ""        # the phone's local date and time
    timezone: str = ""   # the phone's timezone name


def build_system(now, timezone):
    if not now:
        return SYSTEM
    tz = f" (timezone {timezone})" if timezone else ""
    return (
        SYSTEM
        + f"\n\nThe user's current local date and time is {now}{tz}. "
        "Use it to work out words like today, tomorrow, tonight or in 2 hours. "
        "If the time of an alarm or event is unclear, ask."
    )


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatIn, x_app_token: str = Header(default="")):
    if x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    messages = [{"role": "system", "content": build_system(body.now, body.timezone)}]
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
            return {"reply": (msg.content or "").replace("**", ""), "actions": actions}

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
            if c.function.name in DEVICE_TOOLS:
                action, error = prepare_device_action(c.function.name, args)
                if error:
                    output = error
                else:
                    action["id"] = c.id
                    actions.append(action)
                    output = {
                        "status": "prepared",
                        "note": "The app will show the user a Confirm button. "
                                "It is NOT done yet. Tell the user it is ready to confirm.",
                    }
            else:
                output = run_tool(c.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": c.id,
                    "name": c.function.name,
                    "content": json.dumps(output),
                }
            )

    return {"reply": "Sorry, that took too many steps. Please try again.", "actions": actions}


@app.get("/invoices")
def list_invoices(x_app_token: str = Header(default="")):
    if x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        return load_invoices()
    except psycopg.Error as e:
        print("Database error:", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    
