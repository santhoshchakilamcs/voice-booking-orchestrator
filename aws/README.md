# AWS deployment pack — bring your own infra, bring your own key

This project ships no AWS account, no region, no key, and no database
in its code. The user deploying it supplies all of that.

## What the user brings
1. Their own AWS account and region.
2. Their own Deepgram API key, in one of two ways:
   - Per-user, in the browser: paste the key in the demo settings box.
     It is stored only in that browser's localStorage, sent as an
     `X-Deepgram-Key` header, and never written to the database.
   - Or server-side, as an App Runner secret / environment variable
     `DEEPGRAM_API_KEY`, if they want a shared default key.
3. Their own database for production: RDS Postgres, via `DATABASE_URL`.
   SQLite in `data/bookings.db` is local demo only.

## Deploy to App Runner (recommended for live voice)
1. `export AWS_REGION=<your region> AWS_ACCOUNT_ID=<your account>`
2. `./aws/deploy.sh` — builds the Docker image and pushes it to ECR
   in your account.
3. In the AWS console, create an App Runner service from that image,
   port 8000. Add `DEEPGRAM_API_KEY` in secrets only if you want a
   shared default; otherwise leave it empty for bring-your-own-key mode.
4. For production, set `DATABASE_URL` to your RDS instance and do not
   rely on the local SQLite file, which does not persist across
   App Runner restarts in the same way.

## Voice, in order
- With a user key: Deepgram Nova for speech-to-text, Deepgram Aura
  for text-to-speech, via `/api/deepgram/stt` and `/api/deepgram/tts`.
- Without a key: browser speech recognition and speech synthesis.
  Booking, FAQ, availability, and escalation all still work.

## Optional AWS pieces (user-owned, not required to run)
- S3: attach a bucket for call audio/transcripts if the deployer
  wants recordings. Recording is off by default in this demo.
- CloudWatch: App Runner logs flow to CloudWatch automatically.
- Secrets Manager: store the Deepgram key there and reference it
  from App Runner instead of a plain environment variable, if the
  deployer prefers that.

## Local run (no AWS, no key required)
```
.venv/bin/python -m uvicorn app:app --port 8000
# open http://127.0.0.1:8000/
```
