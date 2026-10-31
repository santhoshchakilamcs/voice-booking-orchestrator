# Voice Booking Orchestrator

Conversational AI that books appointments, with ML services orchestrated
end to end. Demo business: Bright Smile Dental Clinic, Austin, TX.
Demo data is synthetic. No real patients, no real payments.

## What it does
- Chat or voice: book, reschedule, cancel, check availability, answer FAQs.
- Orchestrator flow, visible as a trace in the UI:
  1. Urgency check: severe pain, bleeding, swelling, emergency -> human escalation first.
  2. Intent classifier: scikit-learn TF-IDF + LogisticRegression, 9 intents.
  3. Slot extraction: service, date, time from plain English.
  4. FAQ retrieval: TF-IDF over FAQ documents.
  5. Availability service: business hours (Mon-Fri 9-5, Sat 9-1, Sun closed),
     staff conflicts, next-available proposal.
  6. Confirmation step: nothing is written until the user says yes.
  7. Booking write to SQLite, with a reference like BK-1001.
- Low confidence (below 0.20) also escalates to a human, instead of guessing.
  Note: this small training set spreads probability across 9 intents, so
  even exact training phrases score about 0.24-0.34. The threshold is
  tuned for that, and the intent label is still correct.

## Voice: Deepgram, bring your own key
- With a key: Deepgram Nova for speech-to-text, Deepgram Aura for
  text-to-speech, via `/api/deepgram/stt` and `/api/deepgram/tts`.
- How to use your key: paste it in the settings box on the demo page.
  It is stored only in your browser's localStorage, sent as an
  `X-Deepgram-Key` header, and never written to the database.
- Without a key: the page falls back to browser speech recognition
  and speech synthesis. Booking, FAQ, and escalation all still work.
- A server-side default key can be set as `DEEPGRAM_API_KEY`, but it
  is not required, and no key is baked into this code.

## AWS deployment pack: bring your own infra
See `aws/README.md`. In short: the user supplies their own AWS account,
region, and database. The pack gives a Dockerfile, an App Runner config,
and `aws/deploy.sh`, which builds and pushes the image to ECR in the
user's account. App Runner is used because live voice needs a
persistent connection, unlike a pure Lambda setup. For production,
point `DATABASE_URL` at the user's own RDS Postgres. SQLite in
`data/bookings.db` is local demo only.

## Run locally
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app:app --port 8000
# open http://127.0.0.1:8000/
```
Or, after setup: `./run.sh` (port 8001 in this repo's script).

Try these:
- "Book a cleaning tomorrow at 10am" -> then "Yes"
- "What are your hours?"
- "Do you take insurance?"
- "Book a checkup on Friday at 2pm" -> then "Yes"
- "Cancel BK-1001"
- "I have severe pain" -> urgent human escalation

## Verified
Smoke-tested Oct 31, 2026:
- HTTP health, frontend (200), and chat booking flow all passed.
- Direct test: cleaning proposed for Monday Nov 2, 2026 (tomorrow, Nov 1,
  is a Sunday and the clinic is closed), confirmed as BK-1001, then
  cancelled; checkup on Friday Nov 6, 2026 confirmed as BK-1002.
- HTTP test: checkup on Friday Nov 6, 2026 confirmed as BK-1001.
- Without a Deepgram key, `/api/deepgram/tts` correctly returns 501
  with a bring-your-own-key message, and browser-fallback voice works.

## Files
- `app.py` — FastAPI app, orchestrator, ML services, Deepgram client, SQLite.
- `frontend/index.html` — chat, voice, trace, bookings dashboard, key settings.
- `requirements.txt` — fastapi, uvicorn, scikit-learn, python-multipart.
- `Dockerfile`, `apprunner.yaml` — AWS App Runner deployment.
- `aws/deploy.sh`, `aws/README.md` — bring-your-own-infra deployment guide.
- `smoke_test.sh`, `run.sh` — local run and HTTP smoke test.
- `.env.example` — configuration, no real keys.
