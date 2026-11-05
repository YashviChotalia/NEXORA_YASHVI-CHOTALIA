# Anvesh on a Hugging Face Space (Docker SDK): Next.js app + FastAPI API behind one nginx on port 7860.
# One origin keeps the SameSite=Strict session cookie first-party and lets the live call use wss:// on the same host.
# Build context = the folder assembled by scripts/deploy_space.ps1 (backend/, frontend/ and the files next to this one).

# ---- web: Next.js standalone build, API on the same origin --------------------------------------------------------
FROM node:20-bookworm-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
ENV NEXT_PUBLIC_API_URL="" NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---- runtime -------------------------------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 NEXT_TELEMETRY_DISABLED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg nginx libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=web /usr/local/bin/node /usr/local/bin/node

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY backend/requirements.txt backend/requirements-ml.txt ./backend/
# CPU-only torch keeps the image a few GB smaller (Spaces free hardware has no GPU)
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" "torchaudio>=2.2" \
    && pip install -r backend/requirements.txt -r backend/requirements-ml.txt

COPY --chown=user backend/anvesh ./backend/anvesh
COPY --chown=user --from=web /web/.next/standalone ./web
COPY --chown=user --from=web /web/.next/static ./web/.next/static
COPY --chown=user --from=web /web/public ./web/public
COPY --chown=user nginx.conf start.sh ./
RUN chmod +x start.sh

USER user
# Sized for 2 vCPU / 16 GB free hardware: small live ASR, small re-transcription, Indian-language ASR off
# (IndicConformer is gated: set ANVESH_ENABLE_INDIC_ASR=1 and an HF_TOKEN secret to turn it on)
ENV HOME=/home/user ANVESH_DATA_DIR=/home/user/data ANVESH_KEY_DIR=/home/user/.anvesh/keys \
    HF_HOME=/home/user/.cache/huggingface MODELSCOPE_CACHE=/home/user/.cache/modelscope \
    ANVESH_ASR_MODEL=small ANVESH_ASR_FINAL_MODEL=small ANVESH_ENABLE_INDIC_ASR=0

# Bake the model weights into the image so a cold start does not download them (warm-up retries at runtime if this fails)
RUN cd backend && python -m anvesh.download_models --only asr emotion speaker spoof || true

EXPOSE 7860
CMD ["./start.sh"]
