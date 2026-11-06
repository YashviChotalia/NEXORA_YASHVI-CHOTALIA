"""Guided demo calls: a scripted call played through the real analysis, step by step, as if it were live.

What is scripted and what is real:
  scripted  the words (no speech-to-text runs) and the voice clips, which are acted CREMA-D recordings of one speaker
            the emotion head never saw in training (speaker 1002, held out; ODbL, see LICENSES.md)
  real      everything after that: the voice-emotion model and distress track on those clips, field extraction with
            evidence, triage, safety signals, the automatic helpline hand-off (the call counts as live, so the
            Tele-MANAS countdown starts by itself), Ring Linker matches against the training cases, the audit log

Demo cases are marked synthetic, so they live in the Training workspace and never mix with real cases.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

log = logging.getLogger("anvesh.demo")
VOICE = Path(__file__).with_name("data") / "demo" / "voice"
O, C = "officer", "caller"

# (seconds into the call, speaker, words, voice clip kind for the caller: calm / fear / sad / None)
SCENARIOS: dict[str, dict] = {
    "digital_arrest": {
        "title": "Digital arrest, Hinglish (victim)", "language": "hi", "district": "Mumbai",
        "caller": "Demo caller (Mumbai)",
        "steps": [
            (0, O, "Cyber crime helpline, boliye, kya hua?", None),
            (4, C, "Sir, aaj subah mujhe 9876543210 se call aaya, bola CBI se hai, mere Aadhaar se ek parcel "
                   "book hua hai jisme drugs mile hain", "calm"),
            (11, C, "Phir unhone Skype video call pe digital arrest bola, kaha kisi ko mat batana, phone mat katna",
             "fear"),
            (18, O, "Aapne koi paisa bheja?", None),
            (21, C, "Haan sir, maine 85,000 rupaye cbi.verification@ybl pe bheje, HDFC account se, aadha ghanta "
                    "pehle", "fear"),
            (29, C, "Woh abhi bhi video call pe hain sir, please help, mujhe bahut dar lag raha hai", "fear"),
            (36, O, "Aap ghabraiye mat, hum aapke saath hain. Account freeze karwate hain.", None),
            (41, C, "Meri saari savings chali gayi sir, ab mujhe jeene ka mann nahi hai", "sad"),
            (49, O, "Main aapko abhi Tele-MANAS counsellor se jod raha hoon, main line par hi hoon.", None),
        ],
    },
    "tamil_crisis": {
        "title": "Investment fraud, Tamil (victim)", "language": "ta", "district": "Chennai",
        "caller": "Demo caller (Chennai)",
        "steps": [
            (0, O, "Cyber crime helpline, sollunga.", None),
            (4, C, "சார், ஒரு ஆப்பில் முதலீடு செய்தால் இரட்டிப்பு லாபம் என்று சொன்னார்கள்", "calm"),
            (11, C, "நான் ஐம்பதாயிரம் ரூபாய் அனுப்பினேன், இப்போது அவர்கள் போனை எடுக்கவில்லை", "fear"),
            (18, C, "எல்லா பணமும் போய்விட்டது, எனக்கு வாழ விருப்பமில்லை", "sad"),
            (26, O, "Naan ungalai Tele-MANAS counsellor-udan inaikkiren, naan line-il irukkiren.", None),
        ],
    },
}

RUNNING: set[str] = set()


def scenarios() -> list[dict]:
    return [{"key": k, "title": v["title"], "language": v["language"], "steps": len(v["steps"]),
             "seconds": v["steps"][-1][0] + 6} for k, v in SCENARIOS.items()]


def _clips() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {"calm": [], "fear": [], "sad": []}
    tag = {"NEU": "calm", "FEA": "fear", "SAD": "sad"}
    for p in sorted(VOICE.glob("*.wav")):
        k = tag.get(p.stem.split("_")[2] if p.stem.count("_") >= 2 else "")
        if k:
            out[k].append(p)
    return out


def start(case_id: str, key: str, actor: str, speed: float = 1.0) -> None:
    if case_id in RUNNING:
        return
    RUNNING.add(case_id)
    threading.Thread(target=_play, args=(case_id, key, actor, speed), daemon=True).start()


def _play(case_id: str, key: str, actor: str, speed: float) -> None:
    from . import audit, handoff, pipeline
    from .models import SessionLocal
    from .speech import audio as A
    sc = SCENARIOS[key]
    clips, used = _clips(), {"calm": 0, "fear": 0, "sad": 0}
    handoff.LIVE_CASES.add(case_id)  # a demo call is treated as live: automatic hand-off rules apply
    t0 = time.monotonic()
    try:
        with SessionLocal() as db:
            case = pipeline.get_case(db, case_id)
            case.processing = "Demo call in progress: scripted words and acted voice clips, real analysis."
            audit.record(db, actor, "demo.started", case_id, {"scenario": key})
            db.commit()
        for i, (at, speaker, text, voice) in enumerate(sc["steps"]):
            nxt = sc["steps"][i + 1][0] if i + 1 < len(sc["steps"]) else at + 6
            wait = at / speed - (time.monotonic() - t0)
            if wait > 0:
                time.sleep(wait)
            with SessionLocal() as db:
                case = pipeline.get_case(db, case_id)
                if case is None:
                    return
                seg = {"start": float(at), "end": float(nxt) - 0.5, "text": text, "speaker": speaker,
                       "language": sc["language"]}
                if sc["language"] not in ("hi", "en"):
                    seg["text_native"] = text
                pipeline.add_segments(db, case, [seg])
                if voice and clips.get(voice) and case.mode != "suspect_recording":
                    xs = []
                    for _ in range(2):  # two clips per caller line: about 5 s of voice
                        p = clips[voice][used[voice] % len(clips[voice])]
                        used[voice] += 1
                        xs.append(A.decode(p.read_bytes()))
                    import numpy as np
                    pipeline.analyse_windows(case, np.concatenate(xs), offset=float(at))
                db.flush()
                db.refresh(case)
                pipeline.process(db, case, actor)
                db.commit()
    except Exception:  # pragma: no cover - a demo must never take the server down
        log.exception("demo %s failed", case_id)
    finally:
        handoff.LIVE_CASES.discard(case_id)
        RUNNING.discard(case_id)
        with SessionLocal() as db:
            case = pipeline.get_case(db, case_id)
            if case is not None:
                case.processing = None
                audit.record(db, actor, "demo.finished", case_id, {"scenario": key})
                db.commit()
