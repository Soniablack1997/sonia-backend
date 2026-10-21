import hashlib
import hmac
import json
import os
import re
import secrets
import time
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
DATABASE_URL = os.environ["DATABASE_URL"]  # Neon connection string
INVITE_CODE = os.environ.get("INVITE_CODE", "")  # if set, sign-up needs this code
SESSION_DAYS = 90

# Lagos is UTC+1 all year (no daylight saving).
TZ = timezone(timedelta(hours=1))

SYSTEM_BASE = """You are Sonia, a friendly, sharp assistant for small business owners.
Keep replies short and practical. Reply in the language the user writes in
(including Pidgin, Yoruba, Hausa, or Igbo).
If the user writes in Nigerian Pidgin, reply in Nigerian Pidgin, not English.

What you can do: create invoices, do calculations, remember facts about the user,
and prepare phone actions (set an alarm, start a timer, add a calendar event,
draft an SMS, open the phone dialer, draft a WhatsApp message).
You cannot do anything else
yet, so never offer other things such as expense tracking or reading contacts.

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
Messages and calls: you cannot read the user's contacts, so you need the phone number
from the user (or from remembered facts). Never guess a number. Nothing is sent or
called until the user taps Confirm and then sends or dials it themselves. Write the
message text yourself in the user's wording, then use draft_sms or whatsapp_message.
Use call_number to open the dialer.

Memory: when the user tells you something worth keeping (their name, business, usual
currency, customers, prices, preferences) or says "remember", call remember with one
short fact. When they ask you to forget something, call forget with a keyword.
Use the remembered facts below naturally. Never invent facts about the user.

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
    {
        "type": "function",
        "function": {
            "name": "draft_sms",
            "description": "Prepare an SMS draft. The user must tap Confirm and then send it themselves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {"type": "string", "description": "Number, with country code if known"},
                    "message": {"type": "string"},
                },
                "required": ["phone", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "whatsapp_message",
            "description": "Prepare a WhatsApp message draft. The user must tap Confirm and then send it themselves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {"type": "string", "description": "Number, with country code if known"},
                    "message": {"type": "string"},
                },
                "required": ["phone", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_number",
            "description": "Open the phone dialer with a number filled in. The user must tap Confirm and then dial.",
            "parameters": {
                "type": "object",
                "properties": {"phone": {"type": "string"}},
                "required": ["phone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save one short fact about the user for future chats.",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Delete saved facts that contain a keyword.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        },
    },
]

_table_ready = False


def db():
    # prepare_threshold=None keeps this safe with Neon's connection pooler.
    return psycopg.connect(DATABASE_URL, prepare_threshold=None, connect_timeout=10)


def ensure_table():
    global _table_ready
    if _table_ready:
        return
    with db() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "id SERIAL PRIMARY KEY, "
            "email TEXT NOT NULL UNIQUE, "
            "pw_hash TEXT NOT NULL, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "token_hash TEXT PRIMARY KEY, "
            "user_id INT NOT NULL, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS invoices ("
            "id TEXT PRIMARY KEY, "
            "data JSONB NOT NULL, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "id SERIAL PRIMARY KEY, "
            "fact TEXT NOT NULL, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        conn.execute("ALTER TABLE invoices ADD COLUMN IF NOT EXISTS user_id INT")
        conn.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS user_id INT")
    _table_ready = True


# ---------- accounts ----------

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_failures = {}  # email -> list of failure timestamps (simple brute-force guard)


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return salt.hex() + "$" + digest.hex()


def check_password(password, stored):
    try:
        salt_hex, digest_hex = stored.split("$")
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except ValueError:
        return False


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def new_session(conn, user_id):
    token = secrets.token_urlsafe(32)
    conn.execute(
        "INSERT INTO sessions (token_hash, user_id) VALUES (%s, %s)",
        (token_hash(token), user_id),
    )
    return token


def too_many_failures(email):
    recent = [t for t in _failures.get(email, []) if time.time() - t < 900]
    _failures[email] = recent
    return len(recent) >= 5


def current_user(authorization):
    """Return the user id for 'Bearer <token>' or raise 401."""
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Please log in")
    try:
        ensure_table()
        with db() as conn:
            row = conn.execute(
                "SELECT user_id FROM sessions WHERE token_hash = %s "
                "AND created_at > now() - make_interval(days => %s)",
                (token_hash(token), SESSION_DAYS),
            ).fetchone()
    except psycopg.Error as e:
        print("Database error:", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    if not row:
        raise HTTPException(status_code=401, detail="Please log in again")
    return row[0]


# ---------- data (always scoped to one user) ----------

def save_invoice(uid, invoice):
    ensure_table()
    with db() as conn:
        conn.execute(
            "INSERT INTO invoices (id, data, user_id) VALUES (%s, %s, %s)",
            (invoice["id"], Jsonb(invoice), uid),
        )


def load_invoices(uid):
    ensure_table()
    with db() as conn:
        rows = conn.execute(
            "SELECT data FROM invoices WHERE user_id = %s ORDER BY created_at DESC LIMIT 100",
            (uid,),
        ).fetchall()
    return [r[0] for r in rows]


def save_memory(uid, fact):
    ensure_table()
    with db() as conn:
        conn.execute("INSERT INTO memories (fact, user_id) VALUES (%s, %s)", (fact, uid))


def load_memories(uid):
    ensure_table()
    with db() as conn:
        rows = conn.execute(
            "SELECT fact FROM memories WHERE user_id = %s ORDER BY created_at DESC LIMIT 50",
            (uid,),
        ).fetchall()
    return [r[0] for r in rows]


def delete_memories(uid, keyword):
    ensure_table()
    with db() as conn:
        cur = conn.execute(
            "DELETE FROM memories WHERE user_id = %s AND fact ILIKE %s",
            (uid, f"%{keyword}%"),
        )
        return cur.rowcount


def normalize_phone(raw):
    """Return digits with a leading + , or None if it doesn't look like a number."""
    text = str(raw or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if text.startswith("00"):
        digits = digits[2:]
        text = "+" + digits
    if not text.startswith("+"):
        if digits.startswith("0") and len(digits) == 11:
            digits = "234" + digits[1:]  # Nigerian local format
        elif len(digits) == 10 and digits[0] in "789":
            digits = "234" + digits
    if not (8 <= len(digits) <= 15):
        return None
    return "+" + digits


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


def run_tool(name, args, actions, uid):
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
            save_invoice(uid, invoice)
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

        if name in ("draft_sms", "whatsapp_message", "call_number"):
            phone = normalize_phone(args.get("phone"))
            if not phone:
                return {"error": "that phone number looks invalid; ask the user for it"}
            if name == "call_number":
                actions.append(
                    {"type": "call_number", "phone": phone, "summary": f"Open dialer for {phone}"}
                )
            else:
                message = str(args.get("message") or "").strip()[:1000]
                if not message:
                    return {"error": "message is required"}
                kind = "SMS" if name == "draft_sms" else "WhatsApp message"
                actions.append(
                    {
                        "type": name,
                        "phone": phone,
                        "message": message,
                        "summary": f"Draft {kind} to {phone}: {message[:80]}",
                    }
                )
            return {"status": "waiting for the user to tap Confirm and send it themselves"}

        if name == "remember":
            fact = str(args["fact"]).strip()[:300]
            if not fact:
                return {"error": "fact is required"}
            save_memory(uid, fact)
            return {"status": "saved", "fact": fact}

        if name == "forget":
            keyword = str(args["keyword"]).strip()
            if len(keyword) < 2:
                return {"error": "keyword is too short"}
            return {"status": "deleted", "count": delete_memories(uid, keyword)}

        return {"error": "unknown tool"}
    except (KeyError, TypeError, ValueError) as e:
        return {"error": f"bad tool input: {e}"}
    except psycopg.Error as e:
        print("Database error:", e)
        return {"error": "could not save or load that, please try again"}


class Msg(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[Msg]


@app.get("/")
def health():
    return {"status": "ok"}


class AuthIn(BaseModel):
    email: str
    password: str
    invite: str = ""


@app.post("/register")
def register(body: AuthIn):
    email = body.email.strip().lower()
    if not EMAIL_RE.match(email) or len(email) > 200:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    if len(body.password) < 8 or len(body.password) > 200:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if INVITE_CODE and not hmac.compare_digest(body.invite.strip(), INVITE_CODE):
        raise HTTPException(status_code=403, detail="Wrong invite code")
    try:
        ensure_table()
        pw = hash_password(body.password)
        with db() as conn:
            exists = conn.execute("SELECT 1 FROM users WHERE email = %s", (email,)).fetchone()
            if exists:
                raise HTTPException(status_code=409, detail="That email already has an account. Log in instead.")
            first = conn.execute("SELECT count(*) FROM users").fetchone()[0] == 0
            uid = conn.execute(
                "INSERT INTO users (email, pw_hash) VALUES (%s, %s) RETURNING id", (email, pw)
            ).fetchone()[0]
            if first:
                # The very first account adopts data saved before accounts existed.
                conn.execute("UPDATE invoices SET user_id = %s WHERE user_id IS NULL", (uid,))
                conn.execute("UPDATE memories SET user_id = %s WHERE user_id IS NULL", (uid,))
            token = new_session(conn, uid)
    except psycopg.Error as e:
        print("Database error:", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    return {"token": token}


@app.post("/login")
def login(body: AuthIn):
    email = body.email.strip().lower()
    if too_many_failures(email):
        raise HTTPException(status_code=429, detail="Too many tries. Wait 15 minutes and try again.")
    try:
        ensure_table()
        with db() as conn:
            row = conn.execute("SELECT id, pw_hash FROM users WHERE email = %s", (email,)).fetchone()
            ok = bool(row) and check_password(body.password, row[1])
            if not ok:
                _failures.setdefault(email, []).append(time.time())
                raise HTTPException(status_code=401, detail="Wrong email or password")
            token = new_session(conn, row[0])
    except psycopg.Error as e:
        print("Database error:", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    return {"token": token}


@app.post("/logout")
def logout(authorization: str = Header(default="")):
    current_user(authorization)
    token = authorization[7:].strip()
    with db() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = %s", (token_hash(token),))
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatIn, authorization: str = Header(default="")):
    uid = current_user(authorization)

    now = datetime.now(TZ)
    system = SYSTEM_BASE + f"\n\nCurrent date and time in Lagos: {now:%A, %d %B %Y, %H:%M}."
    try:
        facts = load_memories(uid)
    except psycopg.Error as e:
        print("Database error:", e)
        facts = []
    if facts:
        system += "\n\nWhat you remember about the user:\n" + "\n".join("- " + f for f in facts)
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
            output = run_tool(c.function.name, args, actions, uid)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": c.id,
                    "name": c.function.name,
                    "content": json.dumps(output),
                }
            )

    return {"reply": "Sorry, that took too many steps. Please try again.", "actions": []}


@app.get("/memories")
def list_memories(authorization: str = Header(default="")):
    uid = current_user(authorization)
    try:
        return load_memories(uid)
    except psycopg.Error as e:
        print("Database error:", e)
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/invoices")
def list_invoices(authorization: str = Header(default="")):
    uid = current_user(authorization)
    try:
        return load_invoices(uid)
    except psycopg.Error as e:
        print("Database error:", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
