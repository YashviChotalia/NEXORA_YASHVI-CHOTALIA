#!/bin/sh
# Start API, web app and nginx; exit (so the Space restarts) if any of them dies.
set -e
cd /home/user/app

# The browser origin is the Space's own host; the live WebSocket checks it.
export ANVESH_CORS_ORIGINS="${ANVESH_CORS_ORIGINS:-https://${SPACE_HOST:-localhost:7860}}"

(cd backend && python -m anvesh.seed) || echo "seed skipped"

# --proxy-headers trusts X-Forwarded-* only from nginx on 127.0.0.1
(cd backend && exec uvicorn anvesh.api:app --host 127.0.0.1 --port 8000 \
    --proxy-headers --forwarded-allow-ips 127.0.0.1) &
api=$!
(cd web && HOSTNAME=127.0.0.1 PORT=3000 exec node server.js) &
web=$!
nginx -c /home/user/app/nginx.conf -g 'daemon off;' &
proxy=$!

while kill -0 $api 2>/dev/null && kill -0 $web 2>/dev/null && kill -0 $proxy 2>/dev/null; do sleep 5; done
echo "a process exited; stopping" >&2
kill $api $web $proxy 2>/dev/null || true
exit 1
