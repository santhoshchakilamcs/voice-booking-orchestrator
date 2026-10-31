#!/bin/bash
# HTTP smoke test. Kills the uvicorn child pid found via ss, not by pattern.
cd "$(dirname "$0")"
rm -f data/bookings.db
./run.sh > /tmp/vb-http.log 2>&1 &
sleep 6
PID=$(ss -ltnp 2>/dev/null | grep 8001 | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
echo "server pid: $PID"
curl -s http://127.0.0.1:8001/api/health; echo
curl -s -o /dev/null -w "frontend HTTP %{http_code}\n" http://127.0.0.1:8001/
curl -s -X POST http://127.0.0.1:8001/api/chat -H 'Content-Type: application/json' -d '{"session_id":"http1","message":"Book a checkup on Friday at 2pm"}'; echo
curl -s -X POST http://127.0.0.1:8001/api/chat -H 'Content-Type: application/json' -d '{"session_id":"http1","message":"Yes"}'; echo
curl -s http://127.0.0.1:8001/api/bookings; echo
kill "$PID" 2>/dev/null || true
echo "http smoke test done"
