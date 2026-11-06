# Configuration and troubleshooting

## Configuration

All settings are optional environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `ANVESH_ASR_MODEL` | `small` | live speech-to-text size (`base` for low-RAM laptops) |
| `ANVESH_ASR_FINAL_MODEL` | `large-v3-turbo` | accurate re-transcription after a call |
| `ANVESH_EMOTION_MODEL` | `iic/emotion2vec_plus_base` | voice emotion checkpoint |
| `ANVESH_SPOOF_MODEL` | `MelodyMachine/Deepfake-audio-detection-V2` | synthetic-voice detector |
| `ANVESH_ENABLE_ASR_FINAL` / `_EMOTION` / `_SPEAKER` / `_SPOOF` | `1` | turn individual models off (`0`) |
| `ANVESH_TEXT_EMBEDDER` | *(empty)* | e.g. `paraphrase-multilingual-MiniLM-L12-v2` for cross-language script links |
| `ANVESH_VALIDATED_LANGS` | `en,hi` | languages where the voice distress track may score |
| `ANVESH_TELEPHONY` | `demo` | helpline hand-off: `demo` (full flow, no real call), `webhook` (signed POST to your PBX / contact-centre bridge), `off` (suggestions only) |
| `ANVESH_TELEPHONY_URL`, `ANVESH_TELEPHONY_SECRET` | | webhook endpoint and HMAC secret for `webhook` mode |
| `ANVESH_HANDOFF_COUNTDOWN_S` | `10` | seconds the officer has to cancel an automatic transfer |
| `ANVESH_SER_HEAD` | `auto` | telephone-adapted emotion head: `auto` (use it when present), `off` (raw emotion2vec+), or a path to another head file |
| `ANVESH_MASTER_KEY` | generated in `~/.anvesh/keys` | 32-byte base64 master key (use a secret manager in production) |
| `ANVESH_SESSION_IDLE_MIN` / `ANVESH_SESSION_MAX_HOURS` | `30` / `12` | session idle timeout and maximum lifetime |
| `ANVESH_RETENTION_DAYS` | `0` (keep) | erase closed cases older than this at start-up |
| `ANVESH_MAX_UPLOAD_MB` | `100` | largest audio upload accepted |
| `ANVESH_BOOTSTRAP_ADMIN` / `ANVESH_BOOTSTRAP_PASSWORD` | *(empty)* | hosted deployments: create the first administrator at start-up when no account exists (password change forced at first sign-in) |
| `NEXT_PUBLIC_API_URL` (frontend, build time) | `http://localhost:8000` | API address; empty = same origin as the app |

## Troubleshooting

| Problem | Fix |
|---|---|
| A model shows *Error* in the System panel | Run `python -m anvesh.download_models` and check the output; see `logs\backend.log` |
| Slow transcription on an older laptop | Set `ANVESH_ASR_MODEL=base` and `ANVESH_ENABLE_ASR_FINAL=0` |
| Setup page says it is only allowed from the server | Open the app on the same computer as the backend, at `localhost` |
| Port 8000 or 3000 already in use | `start.bat` frees them; otherwise stop the other program |
| Encrypted data cannot be read after moving machines | Copy `~/.anvesh/keys/master.key` with the data (back it up separately) |

Back to the [README](../README.md).
