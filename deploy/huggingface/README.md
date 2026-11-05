---
title: Anvesh
emoji: 🎧
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Real-time speech emotion analysis for cybercrime calls
---

# Anvesh: live demo

Real-time speech emotion and distress signals for cybercrime helpline calls, inside an investigation workspace.
Source code, evaluation and documentation: see the GitHub repository linked from the project README.

Emotion predictions are indicative signals for prioritising calls, never evidence of a person's intentions,
honesty or guilt.

**Open the app at its own address** (`https://<owner>-<space>.hf.space`), not inside the Hugging Face page: the app
refuses to be framed and needs microphone access for live calls.

## Space settings

| Secret / variable | Purpose |
|---|---|
| `ANVESH_BOOTSTRAP_ADMIN` | username of the first administrator (created once, on an empty database) |
| `ANVESH_BOOTSTRAP_PASSWORD` | its password (12+ characters; must be changed at first sign-in) |
| `ANVESH_MASTER_KEY` | optional base64 256-bit key; without it a key is generated per container start |
| `ANVESH_ENABLE_INDIC_ASR` + `HF_TOKEN` | optional: 22 Indian languages (gated model, accept its terms first) |

Storage on free Spaces is not persistent: cases created during a demo are lost when the Space restarts, and the
training scenarios are re-seeded.
