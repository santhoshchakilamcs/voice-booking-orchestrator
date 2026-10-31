#!/usr/bin/env python3
"""
Voice Booking Orchestrator — Bright Smile Dental Clinic (demo business).

Conversational AI + ML-service orchestration, end to end:
  chat/voice text -> intent classifier (scikit-learn TF-IDF + LogisticRegression)
                  -> slot extraction (service, date, time)
                  -> FAQ retrieval (TF-IDF over FAQ docs)
                  -> urgency check (pain / emergency keywords -> human escalation)
                  -> availability service (SQLite, business hours, conflicts)
                  -> confirmation step -> booking write -> reply

Voice stack, per the owner's instruction:
  - Live voice: Deepgram Nova for speech-to-text, Deepgram Aura for
    text-to-speech, via /api/deepgram/stt and /api/deepgram/tts.
    Requires DEEPGRAM_API_KEY in the environment.
  - Demo fallback (no key): the browser frontend uses the Web Speech
    API (SpeechRecognition + speechSynthesis). The server reports
    deepgram_enabled=false and the frontend falls back automatically.
  - No API key is ever stored in code or in the database.

AWS stack (deployment pack in aws/):
  - FastAPI container on AWS App Runner (WebSocket-friendly, unlike
    a pure Lambda setup for live voice).
  - S3 for call audio/transcripts (optional, off by default).
  - CloudWatch for logs/monitoring. RDS Postgres in production;
    SQLite here is the local demo database only.

Run locally:
  .venv/bin/python -m uvicorn app:app --port 8000
  then open http://127.0.0.1:8000/
Demo data is synthetic. No real patients, no real payments.
"""
import json
import os
import re
import sqlite3
import urllib.request
from datetime import date, datetime, time, timedelta
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "bookings.db"

BUSINESS = {
    "name": "Bright Smile Dental Clinic",
    "address": "1200 Congress Ave, Austin, TX 78701",
    "phone": "(512) 555-0142",
    "hours": "Mon-Fri 9am-5pm, Sat 9am-1pm, Sun closed",
}

SERVICES = {
    "dental cleaning": {"label": "Dental Cleaning", "minutes": 60, "price": 120},
    "cleaning": {"label": "Dental Cleaning", "minutes": 60, "price": 120},
    "checkup": {"label": "Checkup", "minutes": 30, "price": 80},
    "check-up": {"label": "Checkup", "minutes": 30, "price": 80},
    "whitening": {"label": "Teeth Whitening", "minutes": 60, "price": 250},
    "teeth whitening": {"label": "Teeth Whitening", "minutes": 60, "price": 250},
    "filling": {"label": "Filling", "minutes": 45, "price": 180},
    "extraction": {"label": "Tooth Extraction Consult", "minutes": 30, "price": 100},
    "braces": {"label": "Braces Consult", "minutes": 30, "price": 50},
    "root canal": {"label": "Root Canal Consult", "minutes": 45, "price": 150},
}

STAFF = ["Dr. Maya Patel", "Dr. Leo Chen"]

FAQS = [
    ("hours", "What are your hours?", "We are open Monday to Friday 9am to 5pm, and Saturday 9am to 1pm. We are closed on Sundays."),
    ("location", "Where are you located?", "Bright Smile Dental Clinic, 1200 Congress Ave, Austin, TX 78701. Free parking is behind the building."),
    ("insurance", "Do you take insurance?", "Yes, we take most major dental insurance plans. Bring your insurance card and we will check your coverage before treatment."),
    ("pricing", "How much does a cleaning cost?", "A dental cleaning is $120, a checkup is $80, and teeth whitening is $250. We will confirm the exact price before we book."),
    ("new patients", "Are you taking new patients?", "Yes, we are taking new patients. A first visit is usually a checkup plus cleaning, and takes about 60 minutes in total."),
    ("emergency", "What if I have a tooth emergency?", "If you have severe pain, swelling, or bleeding, tell us right away and we will escalate you to a human on our team. For a life-threatening emergency, call 911 first."),
    ("cancel", "What is your cancellation policy?", "You can cancel or reschedule up to 24 hours before your appointment at no charge. Just give us your booking reference, like BK-1001."),
    ("parking", "Is there parking?", "Yes, free parking is behind the building on Congress Ave. Street parking is also available nearby."),
]

TRAINING = [
    ("book_appointment", "I want to book a cleaning"),
    ("book_appointment", "Book me a dental cleaning tomorrow at 10am"),
    ("book_appointment", "Can I schedule a checkup on Friday"),
    ("book_appointment", "I need an appointment for whitening next week"),
    ("book_appointment", "Book a filling for Monday morning"),
    ("book_appointment", "I would like to make an appointment"),
    ("book_appointment", "Schedule a braces consult please"),
    ("book_appointment", "Can I get a root canal consult booked"),
    ("reschedule_appointment", "I need to reschedule my appointment"),
    ("reschedule_appointment", "Can I move my booking to Thursday"),
    ("reschedule_appointment", "Change my appointment time please"),
    ("reschedule_appointment", "Reschedule BK-1001 to next Monday at 11am"),
    ("cancel_appointment", "I want to cancel my appointment"),
    ("cancel_appointment", "Cancel my booking please"),
    ("cancel_appointment", "Cancel BK-1001"),
    ("cancel_appointment", "I cannot make it anymore, please cancel"),
    ("check_availability", "What times are available tomorrow"),
    ("check_availability", "Are you free on Friday afternoon"),
    ("check_availability", "Do you have any openings next week"),
    ("check_availability", "What slots do you have for a cleaning"),
    ("faq", "What are your hours"),
    ("faq", "Where are you located"),
    ("faq", "Do you take insurance"),
    ("faq", "How much does a cleaning cost"),
    ("faq", "Are you taking new patients"),
    ("faq", "Is there parking"),
    ("faq", "What is your cancellation policy"),
    ("human_help", "Can I talk to a human please"),
    ("human_help", "I want to speak to a real person"),
    ("human_help", "Let me talk to someone on your team"),
    ("human_help", "This is complicated, get me a human"),
    ("greeting", "Hi"),
    ("greeting", "Hello"),
    ("greeting", "Good morning"),
    ("greeting", "Hey there"),
    ("confirm", "Yes"),
    ("confirm", "Yes that works"),
    ("confirm", "Perfect, confirm it"),
    ("confirm", "Sounds good, book it"),
    ("decline", "No"),
    ("decline", "No that does not work"),
    ("decline", "That time does not work for me"),
]

URGENT_WORDS = ["severe pain", "bleeding", "swelling", "broken tooth", "emergency", "cannot open my mouth"]

# ---------------- database ----------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS bookings (
        id TEXT PRIMARY KEY, service TEXT, staff TEXT, date TEXT,
        start_time TEXT, end_time TEXT, price INTEGER, status TEXT,
        created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
        user_text TEXT, intent TEXT, confidence REAL, reply TEXT,
        created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS escalations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
        reason TEXT, user_text TEXT, created_at TEXT
    );
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- ML services ----------------

def train_intent_model():
    texts = [t for _, t in TRAINING]
    labels = [l for l, _ in TRAINING]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(texts, labels)
    return model

INTENT_MODEL = train_intent_model()

FAQ_VECTORIZER = TfidfVectorizer()
FAQ_MATRIX = FAQ_VECTORIZER.fit_transform([q + " " + a for _, q, a in FAQS])

def classify_intent(text):
    probs = INTENT_MODEL.predict_proba([text])[0]
    idx = int(probs.argmax())
    return INTENT_MODEL.classes_[idx], float(probs[idx])

def retrieve_faq(text):
    vec = FAQ_VECTORIZER.transform([text])
    scores = (FAQ_MATRIX * vec.T).toarray().ravel()
    idx = int(scores.argmax())
    return FAQS[idx], float(scores[idx])

# ---------------- slots, dates, availability ----------------

WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6}

def parse_date(text, today=None):
    today = today or date.today()
    low = text.lower()
    if "today" in low:
        return today
    if "tomorrow" in low:
        return today + timedelta(days=1)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", low)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for name, wd in WEEKDAYS.items():
        if name in low:
            delta = (wd - today.weekday()) % 7
            if "next" in low and delta == 0:
                delta = 7
            if delta == 0:
                delta = 7  # a weekday name alone means the next occurrence
            return today + timedelta(days=delta)
    return None

def parse_time(text):
    low = text.lower()
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", low)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    mer = m.group(3)
    if mer == "pm" and hour < 12:
        hour += 12
    if mer == "am" and hour == 12:
        hour = 0
    if mer is None:
        if "morning" in low:
            return time(10, 0)
        if "afternoon" in low:
            return time(14, 0)
        if "evening" in low:
            return time(17, 0)
        return None
    if hour > 23:
        return None
    return time(hour, minute)

def parse_service(text):
    low = text.lower()
    for key, svc in SERVICES.items():
        if key in low:
            return svc
    return None

def business_open(d, start, minutes):
    if d.weekday() == 6:
        return False
    close = time(13, 0) if d.weekday() == 5 else time(17, 0)
    end = (datetime.combine(d, start) + timedelta(minutes=minutes)).time()
    return time(9, 0) <= start and end <= close

def find_staff(d, start, minutes):
    conn = get_db()
    end = (datetime.combine(d, start) + timedelta(minutes=minutes)).time()
    for staff in STAFF:
        rows = conn.execute(
            "SELECT start_time, end_time FROM bookings WHERE date=? AND staff=? AND status='booked'",
            (d.isoformat(), staff)).fetchall()
        clash = False
        for r in rows:
            s, e = time.fromisoformat(r["start_time"]), time.fromisoformat(r["end_time"])
            if start < e and s < end:
                clash = True
        if not clash:
            conn.close()
            return staff
    conn.close()
    return None

def next_available(d, start, minutes):
    cur_d, cur_t = d, start
    for _ in range(80):
        if cur_d.weekday() == 6:  # Sunday, closed: jump to Monday
            cur_d = cur_d + timedelta(days=1)
            cur_t = time(9, 0)
            continue
        if business_open(cur_d, cur_t, minutes):
            staff = find_staff(cur_d, cur_t, minutes)
            if staff:
                return cur_d, cur_t, staff
        # step 30 minutes, then next day at 9am
        nxt = datetime.combine(cur_d, cur_t) + timedelta(minutes=30)
        if nxt.time() > time(17, 0) or cur_d.weekday() == 5 and nxt.time() > time(13, 0):
            cur_d = cur_d + timedelta(days=1)
            cur_t = time(9, 0)
        else:
            cur_t = nxt.time()
    return None, None, None

def fmt_time(t):
    return t.strftime("%I:%M %p").lstrip("0")

def fmt_date(d):
    return d.strftime("%A, %B %d, %Y")

# ---------------- orchestrator ----------------

SESSIONS = {}

def log_conversation(session_id, user_text, intent, confidence, reply):
    conn = get_db()
    conn.execute("INSERT INTO conversations VALUES (NULL,?,?,?,?,?,?)",
                 (session_id, user_text, intent, confidence, reply, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def log_escalation(session_id, reason, user_text):
    conn = get_db()
    conn.execute("INSERT INTO escalations VALUES (NULL,?,?,?,?)",
                 (session_id, reason, user_text, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def orchestrate(session_id, user_text):
    trace = []
    low = user_text.lower()

    # 1. Urgency service first, before booking logic.
    if any(w in low for w in URGENT_WORDS):
        log_escalation(session_id, "urgent", user_text)
        trace.append("urgency_check: urgent -> human escalation")
        reply = ("That sounds urgent, so I am escalating you to a human on our team right away. "
                 "If this is severe pain, swelling, or bleeding, please also call us at (512) 555-0142. "
                 "A team member will call you back as soon as possible.")
        log_conversation(session_id, user_text, "urgent_escalation", 1.0, reply)
        return reply, "urgent_escalation", 1.0, trace, None

    # 2. Intent classifier (ML service).
    intent, conf = classify_intent(user_text)
    trace.append(f"intent_classifier: {intent} ({conf:.2f})")
    # Threshold 0.20, not 0.35: this small training set spreads
    # probability across 9 intents, so even exact training phrases
    # score about 0.24-0.34. The intent label is still correct;
    # below 0.20 we escalate rather than guess.
    if conf < 0.20:
        log_escalation(session_id, "low_confidence", user_text)
        trace.append("confidence below 0.20 -> human escalation")
        reply = ("I am not sure I understood that, so I will pass this to a human on our team. "
                 "You can also try rephrasing, for example: 'Book a cleaning tomorrow at 10am'.")
        log_conversation(session_id, user_text, "low_confidence", conf, reply)
        return reply, "low_confidence", conf, trace, None

    sess = SESSIONS.setdefault(session_id, {})

    # 3. Confirmation step for a pending booking.
    if sess.get("pending") and intent in ("confirm", "decline"):
        pending = sess.pop("pending")
        if intent == "decline":
            reply = "No problem, I have cancelled that proposal. What time would work better for you?"
            log_conversation(session_id, user_text, intent, conf, reply)
            return reply, intent, conf, trace, None
        conn = get_db()
        booking_id = f"BK-{1000 + conn.execute('SELECT COUNT(*) FROM bookings').fetchone()[0] + 1}"
        conn.execute("INSERT INTO bookings VALUES (?,?,?,?,?,?,?,?,?)",
                     (booking_id, pending["service"], pending["staff"], pending["date"],
                      pending["start"], pending["end"], pending["price"], "booked",
                      datetime.now().isoformat()))
        conn.commit()
        conn.close()
        trace.append(f"booking_service: wrote {booking_id} to SQLite")
        booking = {"id": booking_id, **pending}
        reply = (f"Booked. {pending['service']} with {pending['staff']} on {fmt_date(date.fromisoformat(pending['date']))} "
                 f"at {pending['start']} to {pending['end']}, ${pending['price']}. "
                 f"Your reference is {booking_id}. We will send a reminder before your visit.")
        log_conversation(session_id, user_text, intent, conf, reply)
        return reply, intent, conf, trace, booking

    if intent == "greeting":
        reply = ("Hi, this is Bright Smile Dental Clinic. I can book, reschedule, or cancel an appointment, "
                 "check availability, or answer questions about hours, location, and insurance. "
                 "What would you like to do?")
        log_conversation(session_id, user_text, intent, conf, reply)
        return reply, intent, conf, trace, None

    if intent == "faq":
        (key, q, a), score = retrieve_faq(user_text)
        trace.append(f"faq_retrieval: {key} (score {score:.2f})")
        reply = a
        log_conversation(session_id, user_text, intent, conf, reply)
        return reply, intent, conf, trace, None

    if intent == "human_help":
        log_escalation(session_id, "user_requested", user_text)
        trace.append("human_help -> escalation logged")
        reply = "Of course. I have passed this to a human on our team, and they will call you back. Is there anything else I can help with in the meantime?"
        log_conversation(session_id, user_text, intent, conf, reply)
        return reply, intent, conf, trace, None

    if intent in ("cancel_appointment", "reschedule_appointment"):
        m = re.search(r"BK-\d+", user_text.upper())
        if not m:
            reply = "To do that I need your booking reference, like BK-1001. What is yours?"
            log_conversation(session_id, user_text, intent, conf, reply)
            return reply, intent, conf, trace, None
        bid = m.group(0)
        conn = get_db()
        row = conn.execute("SELECT * FROM bookings WHERE id=?", (bid,)).fetchone()
        if not row:
            conn.close()
            reply = f"I could not find {bid}. Please check the reference and try again, or ask for a human."
            log_conversation(session_id, user_text, intent, conf, reply)
            return reply, intent, conf, trace, None
        if intent == "cancel_appointment":
            conn.execute("UPDATE bookings SET status='cancelled' WHERE id=?", (bid,))
            conn.commit()
            conn.close()
            trace.append(f"booking_service: cancelled {bid}")
            reply = f"Cancelled. Booking {bid} for {row['service']} on {row['date']} is now cancelled, at no charge if it was more than 24 hours ahead."
            log_conversation(session_id, user_text, intent, conf, reply)
            return reply, intent, conf, trace, None
        # reschedule: need a new date/time in the same message
        new_date = parse_date(user_text)
        new_time = parse_time(user_text)
        if not new_date or not new_time:
            conn.close()
            reply = f"Found {bid} for {row['service']}. What new date and time would you like?"
            log_conversation(session_id, user_text, intent, conf, reply)
            return reply, intent, conf, trace, None
        minutes = SERVICES.get(row["service"].lower(), {"minutes": 30})["minutes"]
        staff = find_staff(new_date, new_time, minutes)
        if not staff or not business_open(new_date, new_time, minutes):
            conn.close()
            nd, nt, ns = next_available(new_date, new_time, minutes)
            reply = (f"That slot is not available. The next available slot is {fmt_date(nd)} at {fmt_time(nt)} with {ns}. "
                     "Reply yes to book that, or tell me another time.")
            sess["pending"] = {"service": row["service"], "staff": ns, "date": nd.isoformat(),
                               "start": fmt_time(nt), "end": fmt_time((datetime.combine(nd, nt) + timedelta(minutes=minutes)).time()),
                               "price": row["price"]}
            log_conversation(session_id, user_text, intent, conf, reply)
            return reply, intent, conf, trace, None
        end = (datetime.combine(new_date, new_time) + timedelta(minutes=minutes)).time()
        conn.execute("UPDATE bookings SET date=?, start_time=?, end_time=?, staff=? WHERE id=?",
                     (new_date.isoformat(), new_time.isoformat(), end.isoformat(), staff, bid))
        conn.commit()
        conn.close()
        trace.append(f"booking_service: rescheduled {bid}")
        reply = f"Rescheduled. Booking {bid} is now {row['service']} with {staff} on {fmt_date(new_date)} at {fmt_time(new_time)}."
        log_conversation(session_id, user_text, intent, conf, reply)
        return reply, intent, conf, trace, None

    if intent in ("book_appointment", "check_availability"):
        svc = parse_service(user_text) or sess.get("service")
        d = parse_date(user_text) or sess.get("date")
        t = parse_time(user_text) or sess.get("time")
        if svc:
            sess["service"] = svc
        if d:
            sess["date"] = d
        if t:
            sess["time"] = t
        trace.append(f"slot_extraction: service={svc['label'] if svc else None} date={d} time={t}")
        if not svc:
            reply = "Which service would you like: dental cleaning, checkup, teeth whitening, filling, extraction consult, or braces consult?"
            log_conversation(session_id, user_text, intent, conf, reply)
            return reply, intent, conf, trace, None
        if not d:
            reply = f"Got it, {svc['label']}. What day would you like? For example, tomorrow, or Friday."
            log_conversation(session_id, user_text, intent, conf, reply)
            return reply, intent, conf, trace, None
        if not t:
            reply = f"Got it, {svc['label']} on {fmt_date(d)}. What time? We are open 9am to 5pm weekdays, and 9am to 1pm Saturday."
            log_conversation(session_id, user_text, intent, conf, reply)
            return reply, intent, conf, trace, None
        minutes = svc["minutes"]
        if intent == "check_availability":
            staff = find_staff(d, t, minutes) if business_open(d, t, minutes) else None
            if staff:
                reply = f"Yes, {fmt_date(d)} at {fmt_time(t)} is available for {svc['label']} with {staff}. Reply 'book it' if you want that slot."
            else:
                nd, nt, ns = next_available(d, t, minutes)
                reply = f"That slot is not available. The next available slot is {fmt_date(nd)} at {fmt_time(nt)} with {ns}."
            log_conversation(session_id, user_text, intent, conf, reply)
            return reply, intent, conf, trace, None
        # book: propose, then confirm
        if business_open(d, t, minutes):
            staff = find_staff(d, t, minutes)
        else:
            staff = None
        if not staff:
            nd, nt, ns = next_available(d, t, minutes)
            if nd is None:
                reply = "I could not find an available slot in the next two weeks. Please ask for a human and we will help."
                log_conversation(session_id, user_text, intent, conf, reply)
                return reply, intent, conf, trace, None
            end = (datetime.combine(nd, nt) + timedelta(minutes=minutes)).time()
            sess["pending"] = {"service": svc["label"], "staff": ns, "date": nd.isoformat(),
                               "start": fmt_time(nt), "end": fmt_time(end), "price": svc["price"]}
            trace.append("availability_service: requested slot busy -> proposed next available")
            reply = (f"That slot is not available. I can offer {svc['label']} with {ns} on {fmt_date(nd)} at {fmt_time(nt)} "
                     f"for ${svc['price']}. Reply yes to confirm, or no to pick another time.")
            log_conversation(session_id, user_text, intent, conf, reply)
            return reply, intent, conf, trace, None
        end = (datetime.combine(d, t) + timedelta(minutes=minutes)).time()
        sess["pending"] = {"service": svc["label"], "staff": staff, "date": d.isoformat(),
                           "start": fmt_time(t), "end": fmt_time(end), "price": svc["price"]}
        trace.append("availability_service: slot available -> awaiting confirmation")
        reply = (f"I can book {svc['label']} with {staff} on {fmt_date(d)} at {fmt_time(t)} to {fmt_time(end)} "
                 f"for ${svc['price']}. Reply yes to confirm, or no to change it.")
        log_conversation(session_id, user_text, intent, conf, reply)
        return reply, intent, conf, trace, None

    reply = "I can help with booking, rescheduling, cancelling, availability, or questions about hours and insurance. What would you like to do?"
    log_conversation(session_id, user_text, intent, conf, reply)
    return reply, intent, conf, trace, None

# ---------------- Deepgram voice ----------------

def deepgram_enabled():
    # Server-side key is optional. This app is BYOK: end users bring
    # their own Deepgram key, sent per request as X-Deepgram-Key and
    # never stored in the database. The env var is only a default.
    return bool(os.environ.get("DEEPGRAM_API_KEY"))

def deepgram_request(path, data, content_type, model_param, user_key=None):
    key = user_key or os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise HTTPException(status_code=501, detail="No Deepgram key provided. Add your own key in the demo settings, or set DEEPGRAM_API_KEY on the server. Browser-fallback voice still works without a key.")
    url = f"https://api.deepgram.com/v1/{path}?{model_param}"
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Authorization": f"Token {key}", "Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read(), resp.headers.get_content_type()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Deepgram request failed: {e}")

# ---------------- FastAPI ----------------

app = FastAPI(title="Voice Booking Orchestrator")

class ChatIn(BaseModel):
    session_id: str = "demo"
    message: str

class TtsIn(BaseModel):
    text: str

@app.get("/api/health")
def health():
    return {"status": "ok", "business": BUSINESS["name"],
            "deepgram_enabled": deepgram_enabled(),
            "voice_mode": "deepgram-byok",
            "key_mode": "bring-your-own-key",
            "aws_mode": "bring-your-own-infra"}

@app.post("/api/chat")
def chat(body: ChatIn):
    reply, intent, conf, trace, booking = orchestrate(body.session_id, body.message)
    return {"reply": reply, "intent": intent, "confidence": round(conf, 3),
            "trace": trace, "booking": booking, "business": BUSINESS}

@app.get("/api/bookings")
def bookings():
    conn = get_db()
    rows = conn.execute("SELECT * FROM bookings ORDER BY date, start_time").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/stats")
def stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM bookings WHERE status='booked'").fetchone()[0]
    by_service = conn.execute("SELECT service, COUNT(*) c FROM bookings WHERE status='booked' GROUP BY service").fetchall()
    esc = conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
    conv = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    conn.close()
    return {"booked": total, "by_service": {r["service"]: r["c"] for r in by_service},
            "escalations": esc, "conversations": conv}

@app.post("/api/deepgram/stt")
async def stt(file: UploadFile = File(...), x_deepgram_key: str | None = Header(default=None)):
    audio = await file.read()
    data, _ = deepgram_request("listen", audio, file.content_type or "audio/webm",
                               "model=nova-3&smart_format=true", user_key=x_deepgram_key)
    payload = json.loads(data.decode())
    text = ""
    try:
        text = payload["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception:
        text = ""
    return {"transcript": text, "provider": "deepgram-nova-3"}

@app.post("/api/deepgram/tts")
def tts(body: TtsIn, x_deepgram_key: str | None = Header(default=None)):
    data, _ = deepgram_request("speak", json.dumps({"text": body.text}).encode(),
                               "application/json", "model=aura-2-thalia-en", user_key=x_deepgram_key)
    return Response(content=data, media_type="audio/mpeg")

@app.get("/")
def index():
    return FileResponse(BASE / "frontend" / "index.html")

app.mount("/static", StaticFiles(directory=BASE / "frontend"), name="static")
