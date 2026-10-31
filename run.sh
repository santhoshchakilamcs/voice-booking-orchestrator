#!/bin/bash
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8001
