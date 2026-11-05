"""Case pipeline: audio -> transcript -> facts -> triage -> links, with audit at each step."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .live_ser import adjust_for_role
from . import audit, distress as distress_mod, keys, linker, storage, triage as triage_mod, voice_auth
from .config import settings
from .models import Case, Entity, FieldRow, Recording, Segment
from .nlp import playbooks, safety
from .nlp.extract import Field, Segment as Seg, entities, extract
from .speech import audio as A
from .speech import models as M

log = logging.getLogger("anvesh.pipeline")
PIPELINE_VERSION = "anvesh-pipeline/0.3"
ROLE = {"victim": "victim", "informant": "informant", "suspect_recording": "suspect"}


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


def model_version() -> str:
    parts = [PIPELINE_VERSION]
    for s in M.statuses():
        if s["loaded"]:
            parts.append(f"{s['name']}={s['model']}")
    return "; ".join(parts)


# --- ingest --------------------------------------------------------------------------------

def add_segments(db: Session, case: Case, items: list[dict], recording_id: str | None = None) -> list[Segment]:
    rows = []
    for it in items:
        lp, nsp = it.get("avg_logprob"), it.get("no_speech_prob")
        low = (lp is not None and lp < -1.0) or (nsp is not None and nsp > 0.6)
        row = Segment(id=new_id("SG"), case_id=case.id, recording_id=recording_id, start=float(it["start"]),
                      end=float(it["end"]), text=it["text"].strip(), text_native=it.get("text_native"),
                      speaker=it.get("speaker"),
                      language=it.get("language"), avg_logprob=lp, no_speech_prob=nsp, low_confidence=low,
                      words=it.get("words"))
        db.add(row)
        rows.append(row)
    db.flush()
    return rows


def transcribe_and_analyse(x: np.ndarray, mode: str, kind: str, offset: float = 0.0,
                           transcribe: bool = True, analyse: bool = True, previous_text: str = "",
                           model: "M.ASR | None" = None, language: str | None = None) -> dict:
    """All slow model work, done WITHOUT holding a database transaction."""
    out: dict = {"segments": [], "language": None, "windows": [], "embedding": None}
    if transcribe:
        segs, lang = (model or M.asr).transcribe(x, offset=offset, previous_text=previous_text,
                                                 language=language if language not in (None, "", "auto") else None)
        speaker = "suspect" if kind == "suspect_audio" else None
        for s in segs:
            s["language"] = lang
            s["speaker"] = speaker
        out["segments"], out["language"] = segs, lang
    if analyse and mode != "suspect_recording":
        out["windows"] = compute_windows(x, offset, out["segments"], ROLE.get(mode, "victim"))
    if kind == "suspect_audio":
        out["embedding"] = M.speaker.embed(x)
    if analyse:
        out["voice_auth"] = check_voice(x, mode)
    return out


def check_voice(x: np.ndarray, mode: str) -> dict:
    """Synthetic / cloned voice check (model when loaded, acoustic cues always). Indicative only."""
    try:
        p = M.spoof.p_fake(x)
    except Exception as e:  # pragma: no cover - a model failure must not break ingestion
        log.warning("synthetic-voice model failed: %s", e)
        p = None
    return voice_auth.assess(x, p, M.spoof.model_id, mode)


VERDICT_RANK = {"likely_synthetic": 4, "needs_verification": 3, "uncertain": 2, "likely_real": 1,
                "insufficient_audio": 0, "not_checked": 0}


def summarise_voice(case: Case) -> None:
    """Case-level verdict = the most concerning checked recording (a clone in any part of the call matters)."""
    checked = [r.voice_auth for r in case.recordings if r.voice_auth]
    if checked:
        worst = max(checked, key=lambda v: (VERDICT_RANK.get(v.get("verdict"), 0), v.get("score") or 0))
        case.voice_auth = {**worst, "recordings_checked": len(checked)}


def ingest_audio(db: Session, case: Case, data: bytes, filename: str, mime: str, kind: str, actor: str,
                 offset: float = 0.0, transcribe: bool = True, analyse: bool = True,
                 precomputed: dict | None = None) -> Recording:
    digest = audit.sha256_bytes(data)
    x = A.decode(data)
    res = precomputed if precomputed is not None else transcribe_and_analyse(
        x, case.mode, kind, offset, transcribe, analyse)
    path = storage.save(digest, data)
    rec = Recording(id=new_id("RC"), case_id=case.id, kind=kind, filename=filename, mime=mime or "audio/wav",
                    sha256=digest, size_bytes=len(data), stored_path=path, duration=round(len(x) / A.SR, 2),
                    speaker_embedding=res.get("embedding"), offset=round(offset, 3), voice_auth=res.get("voice_auth"))
    db.add(rec)
    db.flush()
    audit.record(db, actor, "recording.ingested", case.id,
                 {"recording_id": rec.id, "sha256": digest, "bytes": len(data), "filename": filename, "kind": kind},
                 model_version())
    if res.get("segments"):
        add_segments(db, case, res["segments"], rec.id)
        case.language = case.language or res.get("language")
    if res.get("windows"):
        case.windows = list(case.windows or []) + res["windows"]
    if rec.voice_auth:
        if rec not in case.recordings:
            case.recordings.append(rec)
        summarise_voice(case)
        db.flush()  # persist before callers refresh the case
        if rec.voice_auth.get("verdict") in ("likely_synthetic", "needs_verification"):
            audit.record(db, actor, "voice.flagged", case.id, {"recording_id": rec.id, "verdict": rec.voice_auth["verdict"],
                                                              "score": rec.voice_auth.get("score")})
    return rec


MIN_VOICED = 0.3     # fraction of frames with speech energy
MIN_RMS = 0.006      # below this the window is treated as silence


def _utterance_spans(x: np.ndarray, offset: float, segs: list[dict]) -> list[tuple[float, float]]:
    """Emotion models are trained on utterances, so score per spoken line (split if > 8 s)."""
    spans = []
    total = len(x) / A.SR
    for sg in segs:
        a, b = max(sg["start"] - offset, 0.0), min(sg["end"] - offset, total)
        while b - a > 8.0:
            spans.append((a, a + 6.0))
            a += 6.0
        if b - a >= 0.8:
            spans.append((a, b))
    return spans


def _positive(v) -> bool:
    try:
        return float(v or 0) > 0
    except (TypeError, ValueError):
        return False


def compute_windows(x: np.ndarray, offset: float = 0.0, segs: list[dict] | None = None,
                    role: str = "victim") -> list[dict]:
    """Voice windows for the distress track: 3 s windows, silence skipped.

    The same windows as a live call (live_ser.WINDOW_S) and as the evaluation (eval/run_call_eval.py), so an
    uploaded recording is scored exactly the way the measured system was. `segs` is accepted for compatibility.
    """
    spans = [(t0, t1) for t0, t1, _ in A.windows(x, 3.0)]
    out = []
    for t0, t1 in spans:
        chunk = x[int(t0 * A.SR):int(t1 * A.SR)]
        feats = A.features(chunk)
        voiced = 1 - feats.get("pause_ratio", 1.0) if feats else 0.0
        silent = not feats or feats.get("rms", 0.0) < MIN_RMS or voiced < MIN_VOICED
        emo = None
        if not silent:
            try:
                emo = adjust_for_role(M.emotion.predict(chunk), role)
            except Exception as e:  # pragma: no cover
                log.warning("emotion inference failed: %s", e)
        out.append({"t_start": round(t0 + offset, 2), "t_end": round(t1 + offset, 2), "emotions": emo,
                    "acoustic": feats, "silent": silent})
    return out


def analyse_windows(case: Case, x: np.ndarray, offset: float = 0.0) -> None:
    if case.mode == "suspect_recording":
        return
    case.windows = list(case.windows or []) + compute_windows(x, offset)


# --- analysis ------------------------------------------------------------------------------

def _segs(case: Case) -> list[Seg]:
    return [Seg(s.id, s.start, s.end, s.text, s.speaker, s.words) for s in case.segments]


def process(db: Session, case: Case, actor: str = "system", now: datetime | None = None) -> Case:
    now = now or datetime.now()
    segs = _segs(case)
    role = ROLE[case.mode]

    # narrative = what the caller / suspect said (officer lines excluded)
    caller = [s for s in segs if s.speaker != "officer"]
    case.narrative = " ".join(s.text for s in caller)

    # 1) facts with evidence; keep officer-verified or corrected values
    found = extract(caller, case.call_started_at)
    existing = {f.name: f for f in case.fields}
    for name, fld in found.items():
        row = existing.get(name)
        if row and row.status in ("verified", "corrected"):
            continue
        ev = [e.__dict__ for e in fld.evidence]
        if isinstance(fld.value, float) and fld.value.is_integer():
            fld.value = int(fld.value)
        if row:
            row.value, row.display, row.confidence, row.evidence, row.note = (
                str(fld.value), fld.display, fld.confidence, ev, fld.note)
            row.model_value = str(fld.value)
        else:
            db.add(FieldRow(case_id=case.id, name=name, value=str(fld.value), display=fld.display,
                            confidence=fld.confidence, status="predicted", evidence=ev, note=fld.note,
                            model_value=str(fld.value)))
    db.flush()
    db.refresh(case)
    current = {f.name: Field(f.name, f.value, f.display, f.confidence, [], f.status) for f in case.fields}

    # 2) entities for linking (from current values, including corrections)
    db.execute(delete(Entity).where(Entity.case_id == case.id))
    for typ, val in entities(current):
        db.add(Entity(case_id=case.id, type=typ, value=val, value_hash=keys.blind_index(typ, val)))

    # 3) playbook + co-pilot (victim / informant) or tactics (suspect)
    pb, conf, cues = playbooks.classify(case.narrative)
    case.playbook_key, case.playbook_conf = pb.key, conf
    if role == "suspect":
        hits = playbooks.tactics_in(case.narrative)
        case.tactics = {"items": [{"key": k, "label": playbooks.TACTIC_LABELS[k], "cues": v} for k, v in hits.items()],
                        "note": "Investigative lead from the recording's words; not a finding of guilt."}
        case.copilot = []
    else:
        case.tactics = {}
        present = set(current)
        case.copilot = playbooks.copilot_prompts(pb, present, set(case.asked or []))

    # 4) human-safety signals (victim / informant only)
    if role == "suspect":
        sig = safety.SafetySignals()
    else:
        sig = safety.scan(caller)
    coercion = len(sig.coercion_hits) > 0
    crisis = len(sig.crisis_hits) > 0
    case.safety = {
        "coercion": coercion, "crisis": crisis,
        "coercion_hits": sig.coercion_hits, "crisis_hits": sig.crisis_hits,
        "coercion_message": ("The caller may still be under the scammer's control. Ask them to end that call now; "
                             "police do not arrest anyone over a video call.") if coercion else None,
        "crisis_message": safety.CRISIS_RESOURCE if crisis else None,
        "playbook_advice": pb.advice,
    }

    # 5) distress track (role-gated, abstains when unsure)
    windows = [distress_mod.Window(w["t_start"], w["t_end"], w.get("emotions"), w.get("acoustic"), w.get("silent", False))
               for w in (case.windows or [])]
    loss = current.get("amount_lost")
    d = distress_mod.assess(role, windows, sig.distress_lexical, sig.crisis_hits, case.language,
                            set(settings.validated_languages), coercion=bool(sig.coercion_hits),
                            loss_reported=_positive(loss.value if loss else None))
    case.distress = d.to_dict()

    # 6) triage
    verified = {f.name for f in case.fields if f.status in ("verified", "corrected")}
    call_age = (now - case.call_started_at).total_seconds() / 60
    t = triage_mod.assess(case.mode, current, now, coercion, crisis, d.level if d.enabled else None, verified,
                          call_age)
    case.triage = t.to_dict()
    case.priority, case.priority_score = t.priority, t.score

    # 7) links
    db.flush()
    db.refresh(case)
    n_links = linker.link_case(db, case)

    audit.record(db, actor, "case.analysed", case.id,
                 {"fields": sorted(current), "priority": t.priority, "score": t.score, "links": n_links,
                  "playbook": pb.key, "coercion": coercion, "crisis": crisis,
                  "distress": d.level}, model_version())
    return case


# --- read-back -----------------------------------------------------------------------------

def readback(case: Case, lang: str = "hi") -> str:
    f = {x.name: x for x in case.fields}
    parts = []
    if lang == "en":
        if "amount_lost" in f:
            parts.append(f"You lost {f['amount_lost'].display.replace('₹', 'rupees ')}")
        if "incident_time" in f:
            parts.append(f"at {f['incident_time'].display.split(' (')[0]}")
        if "upi_id" in f:
            parts.append(f"sent to {f['upi_id'].value.replace('@', ' at ')}")
        if "bank" in f:
            parts.append(f"from your {f['bank'].display} account")
        return (" ".join(parts) + ". Is that correct?") if parts else "I have not captured the details yet."
    if "amount_lost" in f:
        parts.append(f"Aapke {f['amount_lost'].display.replace('₹', '')} rupaye gaye")
    if "incident_time" in f:
        parts.append(f"{f['incident_time'].display.split(' (')[0].split(', ')[-1]} baje")
    if "upi_id" in f:
        parts.append(f"{f['upi_id'].value.replace('@', ' at the rate ')} ko")
    if "bank" in f:
        parts.append(f"{f['bank'].display} khaate se")
    return (", ".join(parts) + ". Kya yeh sahi hai?") if parts else "Abhi tak details record nahi hui hain."


def get_case(db: Session, case_id: str) -> Case | None:
    return db.execute(select(Case).where(Case.id == case_id)).scalar_one_or_none()


# --- accurate second pass ---------------------------------------------------------------------

def refine_recording(case_id: str, rec_id: str, actor: str = "system") -> None:
    """Re-transcribe a whole recording with the accurate model and rebuild that part of the case.

    Runs in a background thread after a live call ends (and after uploads when the accurate model was
    not ready yet). The live draft stays visible until the refined version replaces it.
    """
    from .models import SessionLocal
    model = M.asr_final if M.asr_final.enabled else M.asr
    with SessionLocal() as db:
        case = get_case(db, case_id)
        rec = db.get(Recording, rec_id)
        if not case or not rec:
            return
        case.processing = f"Refining transcript with {model.model_id}…"
        db.commit()
        data = storage.load(rec.stored_path)
        offset, kind, mode, lang = rec.offset or 0.0, rec.kind, case.mode, case.language
    try:
        x = A.decode(data)
        res = transcribe_and_analyse(x, mode, kind, offset, model=model, language=lang)
    except Exception as e:  # pragma: no cover - model failure keeps the live draft
        log.warning("refine failed: %s", e)
        with SessionLocal() as db:
            case = get_case(db, case_id)
            case.processing = None
            db.commit()
        return
    end = offset + len(x) / A.SR
    with SessionLocal() as db:
        case = get_case(db, case_id)
        old = [s for s in case.segments if s.recording_id == rec_id]
        new_chars = sum(len(s["text"]) for s in res["segments"])
        if old and new_chars < 0.4 * sum(len(s.text or "") for s in old):
            # the accurate pass recovered much less speech than the live draft: keep the draft
            log.warning("refine of %s kept the live draft (%d vs %d characters)", rec_id, new_chars,
                        sum(len(s.text or "") for s in old))
            case.processing = None
            db.commit()
            return
        labels = {round(s.start): s.speaker for s in old if s.speaker}
        for s in old:
            db.delete(s)
        db.flush()
        for sg in res["segments"]:
            sg["speaker"] = sg.get("speaker") or labels.get(round(sg["start"]))
        add_segments(db, case, res["segments"], rec_id)
        keep = [w for w in (case.windows or []) if not (offset - 0.01 <= w["t_start"] < end)]
        case.windows = keep + res["windows"]
        case.processing = None
        db.flush()
        db.refresh(case)
        audit.record(db, actor, "transcript.refined", case.id,
                     {"recording_id": rec_id, "model": model.model_id, "segments": len(res["segments"])},
                     model_version())
        process(db, case, actor)
        db.commit()


def refine_async(case_id: str, rec_id: str, actor: str = "system") -> None:
    import threading
    threading.Thread(target=refine_recording, args=(case_id, rec_id, actor), daemon=True).start()
