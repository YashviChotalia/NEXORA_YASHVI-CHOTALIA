"""Anvesh HTTP + WebSocket API."""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Path, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field as PField
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import audit, auth, linker, live_ser, official, pipeline, retention, storage
from .auth import Principal
from .security_http import MAX_UPLOAD_BYTES, SecurityMiddleware
from .config import settings
from .models import AuditEvent, AuthSession, Case, FieldRow, Link, Recording, Segment, SessionLocal, User, init_db, utcnow
from .nlp import playbooks
from .speech import audio as A
from .speech import models as M
from .triage import FIELD_LABELS

log = logging.getLogger("anvesh.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

@asynccontextmanager
async def lifespan(_app):
    init_db()
    storage.rekey_legacy()
    retention.purge_expired()
    M.warmup()  # load speech models in the background so the first live call does not wait
    yield


init_db()
app = FastAPI(title="Anvesh API", version="0.3.0", lifespan=lifespan,
              description="Role-aware cybercrime call intelligence: call → case file → linked rings.")
app.add_middleware(SecurityMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=True,
                   allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Content-Type", "X-CSRF-Token"],
                   expose_headers=["X-Evidence-SHA256"], max_age=600)


db_session = auth.db_dep  # shared with the auth dependency (one DB session per request)

CASE_READ = auth.require("case:read")
CASE_WRITE = auth.require("case:write")
CaseId = Path(pattern=r"^AV-\d{4}-[A-Z0-9]{4,6}$")
ObjId = Path(pattern=r"^[A-Za-z0-9-]{1,40}$")
KeyName = Path(pattern=r"^[a-z0-9_]{1,40}$")
_viewed: dict[tuple[str, str], float] = {}
MAX_LIVE_MINUTES = int(os.getenv("ANVESH_MAX_LIVE_MINUTES", "120"))
AUDIO_TYPES = ("audio/", "video/webm", "video/mp4", "application/octet-stream")


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    out = bytearray()
    while chunk := await file.read(1024 * 1024):
        out.extend(chunk)
        if len(out) > limit:
            raise HTTPException(413, f"File too large (limit {limit // (1024 * 1024)} MB)")
    return bytes(out)


def _safe_filename(name: str | None) -> str:
    base = re.split(r"[\\/]", name or "upload")[-1]
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")[:120]
    return base or "upload"


def _finite(o):
    """Replace NaN/Infinity (invalid JSON for browsers) with None, recursively."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def _can_reveal(p: Principal | None, case: Case) -> bool:
    if case.mode != "informant":
        return True
    return p is not None and (p.can("informant:reveal") or p.username == case.created_by)


# --- serialisation ----------------------------------------------------------------------------

def _field_json(f: FieldRow) -> dict:
    return {"name": f.name, "label": FIELD_LABELS.get(f.name, f.name), "value": f.value, "display": f.display,
            "confidence": f.confidence, "status": f.status, "evidence": f.evidence, "note": f.note,
            "model_value": f.model_value, "updated_by": f.updated_by}


def _amount(case: Case) -> float:
    for f in case.fields:
        if f.name == "amount_lost":
            try:
                return float(f.value)
            except ValueError:
                return 0.0
    return 0.0


def case_summary(case: Case, p: Principal | None = None) -> dict:
    pb = playbooks.PLAYBOOK_BY_KEY.get(case.playbook_key or "")
    reveal = _can_reveal(p, case)
    return {
        "id": case.id, "created_at": case.created_at.isoformat() + "Z", "mode": case.mode, "channel": case.channel,
        "status": case.status, "caller_label": case.caller_label if reveal else "Protected informant",
        "identity_protected": not reveal, "district": case.district,
        "priority": case.priority, "priority_score": case.priority_score, "synthetic": case.synthetic,
        "playbook": {"key": case.playbook_key, "label": pb.label if pb else "Type unclear", "confidence": case.playbook_conf},
        "amount": _amount(case), "headline": (case.triage or {}).get("headline"),
        "coercion": bool((case.safety or {}).get("coercion")), "crisis": bool((case.safety or {}).get("crisis")),
        "voice_flag": (case.voice_auth or {}).get("verdict") in ("likely_synthetic", "needs_verification"),
        "language": case.language, "intake_seconds": case.intake_seconds,
    }


def case_detail(db: Session, case: Case, p: Principal | None = None) -> dict:
    links = linker.case_links(db, case.id)
    other_ids = {l.case_b if l.case_a == case.id else l.case_a for l in links}
    others = {c.id: c for c in db.execute(select(Case).where(Case.id.in_(other_ids))).scalars()} if other_ids else {}
    d = case_summary(case, p)
    d.update({
        "call_started_at": case.call_started_at.isoformat() + "Z",
        "processing": case.processing,
        "segments": [{"id": s.id, "start": s.start, "end": s.end, "text": s.text, "text_native": s.text_native,
                      "speaker": s.speaker,
                      "language": s.language, "low_confidence": s.low_confidence, "recording_id": s.recording_id}
                     for s in case.segments],
        "fields": sorted((_field_json(f) for f in case.fields),
                         key=lambda f: list(FIELD_LABELS).index(f["name"]) if f["name"] in FIELD_LABELS else 99),
        "triage": case.triage, "distress": case.distress, "safety": case.safety, "tactics": case.tactics,
        "copilot": case.copilot,
        "local_context": official.context_for(case.district),
        "official_scam_type": official.playbook_context(case.playbook_key),
        "scenario": case.scenario,
        "voice_auth": case.voice_auth,
        "recordings": [{"id": r.id, "kind": r.kind, "filename": r.filename, "sha256": r.sha256,
                        "duration": r.duration, "size_bytes": r.size_bytes, "offset": r.offset or 0.0,
                        "voice_auth": r.voice_auth,
                        "has_voiceprint": bool(r.speaker_embedding)} for r in case.recordings],
        "links": [{"id": l.id, "kind": l.kind, "score": l.score, "status": l.status, "detail": l.detail,
                   "other": case_summary(others[o], p) if (o := (l.case_b if l.case_a == case.id else l.case_a)) in others else {"id": o}}
                  for l in links],
    })
    return _finite(d)


def _get(db: Session, case_id: str) -> Case:
    c = pipeline.get_case(db, case_id)
    if not c:
        raise HTTPException(404, "Case not found")
    return c


# --- schemas ----------------------------------------------------------------------------------

class CaseIn(BaseModel):
    mode: str = PField(pattern="^(victim|suspect_recording|informant)$")
    channel: str = PField(default="helpline", pattern="^(helpline|upload|informant_line|walk_in)$")
    caller_label: str = PField(default="Unnamed caller", max_length=120)
    district: str | None = PField(default=None, max_length=80)
    call_started_at: datetime | None = None
    synthetic: bool = False
    # caller's language if the officer knows it ("auto" or empty = detect); ISO / IndicConformer code
    language: str | None = PField(default=None, pattern="^(auto|[a-z]{2,3})$")


class CasePatch(BaseModel):
    status: str | None = PField(default=None, pattern="^(open|review|closed)$")
    mode: str | None = PField(default=None, pattern="^(victim|suspect_recording|informant)$")
    district: str | None = PField(default=None, max_length=80)
    caller_label: str | None = PField(default=None, max_length=120)


class SegmentIn(BaseModel):
    start: float = PField(ge=0, le=86400)
    end: float = PField(ge=0, le=86400)
    text: str = PField(min_length=1, max_length=4000)
    speaker: str | None = PField(default=None, pattern="^(caller|officer|suspect|informant|unknown)$")


class SegmentsIn(BaseModel):
    segments: list[SegmentIn] = PField(max_length=500)


class FieldAction(BaseModel):
    action: str = PField(pattern="^(verify|correct|reject|reported)$")
    value: str | None = PField(default=None, max_length=500)
    display: str | None = PField(default=None, max_length=500)


class SpeakerPatch(BaseModel):
    speaker: str = PField(pattern="^(caller|officer|suspect|informant|unknown)$")


class LinkPatch(BaseModel):
    status: str = PField(pattern="^(confirmed|rejected|candidate)$")


# --- routes -----------------------------------------------------------------------------------

@app.get("/api/health")
def health(request: Request, db: Session = Depends(db_session)):
    db.execute(select(func.count(Case.id))).scalar()
    out = {"ok": True}
    if auth.principal_from_token(db, request.cookies.get(auth.COOKIE)):  # details only for signed-in users
        out.update({"version": app.version, "pipeline": pipeline.PIPELINE_VERSION, "models": M.statuses(),
                    "validated_languages": list(settings.validated_languages),
                    "live_latency": live_ser.LATENCY.summary()})
    return out


# --- authentication ---------------------------------------------------------------------------

class LoginIn(BaseModel):
    username: str = PField(min_length=1, max_length=64)
    password: str = PField(min_length=1, max_length=128)


class SetupIn(BaseModel):
    username: str = PField(min_length=3, max_length=32)
    display_name: str = PField(min_length=1, max_length=120)
    password: str = PField(min_length=1, max_length=128)


class PasswordIn(BaseModel):
    current: str = PField(min_length=1, max_length=128)
    new: str = PField(min_length=1, max_length=128)


class UserIn(BaseModel):
    username: str = PField(min_length=3, max_length=32)
    display_name: str = PField(min_length=1, max_length=120)
    roles: list[str] = PField(min_length=1, max_length=3)
    password: str = PField(min_length=1, max_length=128)


class UserPatch(BaseModel):
    display_name: str | None = PField(default=None, max_length=120)
    roles: list[str] | None = PField(default=None, max_length=3)
    disabled: bool | None = None
    password: str | None = PField(default=None, max_length=128)


def _user_json(u: User) -> dict:
    return {"id": u.id, "username": u.username, "display_name": u.display_name, "roles": u.roles or [],
            "disabled": u.disabled, "must_change_password": u.must_change_password,
            "locked": auth.is_locked(u), "last_login": auth.utc_iso(u.last_login), "created_at": auth.utc_iso(u.created_at)}


def _me(p: Principal, csrf: str | None = None) -> dict:
    d = {"user": {"username": p.username, "display_name": p.display_name, "roles": p.roles,
                  "permissions": sorted(p.perms), "must_change_password": p.must_change_password}}
    if csrf:
        d["csrf_token"] = csrf
    return d


def _check_user_input(username: str, roles: list[str] | None, password: str | None):
    if not auth.USERNAME_RE.match(username):
        raise HTTPException(422, "Username: 3-32 lowercase letters, digits, dot, dash or underscore")
    if roles is not None and (not roles or any(r not in auth.ROLES for r in roles)):
        raise HTTPException(422, f"Roles must be from {', '.join(auth.ROLES)}")
    if password is not None and (problems := auth.password_problems(password, username)):
        raise HTTPException(422, "Password needs " + "; ".join(problems))


@app.get("/api/auth/state")
def auth_state(request: Request, db: Session = Depends(db_session)):
    p = auth.principal_from_token(db, request.cookies.get(auth.COOKIE))
    out = {"setup_required": auth.user_count(db) == 0, "signed_in": p is not None}
    if p:
        out.update(_me(p, auth.rotate_csrf(db, p)))
    return out


@app.post("/api/auth/setup", status_code=201)
def auth_setup(body: SetupIn, request: Request, response: Response, db: Session = Depends(db_session)):
    """First run only: create the first administrator. Allowed only from this computer."""
    if auth.user_count(db) > 0:
        raise HTTPException(409, "Setup has already been completed")
    if auth.client_key(request) not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(403, "First-run setup is only allowed from the server itself")
    username = body.username.strip().lower()
    _check_user_input(username, ["admin", "supervisor"], body.password)
    u = User(username=username, display_name=body.display_name.strip(), roles=["admin", "supervisor"],
             pw_hash=auth.hash_password(body.password), last_login=utcnow())
    db.add(u)
    db.flush()
    audit.record(db, u.username, "auth.setup", None, {"roles": u.roles})
    csrf = auth.create_session(db, u, response, request)
    return _me(auth.Principal(u.id, u.username, u.display_name, u.roles, auth.perms_for(u.roles), ""), csrf)


@app.post("/api/auth/login")
def auth_login(body: LoginIn, request: Request, response: Response, db: Session = Depends(db_session)):
    ip = auth.client_key(request)
    if not auth.limiter.allow(f"login:{ip}", 10, 60):
        raise HTTPException(429, "Too many sign-in attempts. Wait a minute and try again.")
    username = body.username.strip().lower()
    u = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if u and auth.is_locked(u):
        audit.record(db, username, "auth.login_blocked", None, {"reason": "locked"})
        db.commit()
        raise HTTPException(423, "Account temporarily locked after repeated failed sign-ins. Try again later.")
    ok = auth.verify_password(body.password, u.pw_hash if u else auth._dummy_hash())  # constant work
    if not ok or u.disabled:
        auth.register_failure(u)
        audit.record(db, username if u else "unknown", "auth.login_failed", None, {})
        db.commit()
        raise HTTPException(401, "Incorrect username or password")
    u.failed_attempts = 0
    u.locked_until = None
    u.last_login = utcnow()
    csrf = auth.create_session(db, u, response, request)
    audit.record(db, u.username, "auth.login", None, {})
    return _me(auth.Principal(u.id, u.username, u.display_name, list(u.roles), auth.perms_for(u.roles), "",
                              u.must_change_password), csrf)


@app.post("/api/auth/logout")
def auth_logout(response: Response, db: Session = Depends(db_session), p: Principal = Depends(auth.current)):
    s = db.get(AuthSession, p.session_hash)
    s.revoked = True
    response.delete_cookie(auth.COOKIE, path="/")
    audit.record(db, p.username, "auth.logout", None, {})
    return {"ok": True}


@app.post("/api/auth/password")
def auth_password(body: PasswordIn, db: Session = Depends(db_session), p: Principal = Depends(auth.current)):
    u = db.get(User, p.id)
    if not auth.verify_password(body.current, u.pw_hash):
        raise HTTPException(403, "Current password is incorrect")
    _check_user_input(u.username, None, body.new)
    u.pw_hash = auth.hash_password(body.new)
    u.must_change_password = False
    auth.revoke_user_sessions(db, u.id, keep=p.session_hash)
    audit.record(db, p.username, "auth.password_changed", None, {})
    return {"ok": True}


@app.get("/api/users")
def list_users(db: Session = Depends(db_session), p: Principal = Depends(auth.require("user:admin"))):
    return [_user_json(u) for u in db.execute(select(User).order_by(User.id)).scalars()]


@app.post("/api/users", status_code=201)
def create_user(body: UserIn, db: Session = Depends(db_session), p: Principal = Depends(auth.require("user:admin"))):
    username = body.username.strip().lower()
    _check_user_input(username, body.roles, body.password)
    if db.execute(select(User).where(User.username == username)).scalar_one_or_none():
        raise HTTPException(409, "That username is taken")
    u = User(username=username, display_name=body.display_name.strip(), roles=sorted(set(body.roles)),
             pw_hash=auth.hash_password(body.password), must_change_password=True)
    db.add(u)
    db.flush()
    audit.record(db, p.actor, "user.created", None, {"user": username, "roles": u.roles})
    return _user_json(u)


@app.patch("/api/users/{user_id}")
def update_user(body: UserPatch, user_id: Annotated[int, Path(ge=1)], db: Session = Depends(db_session),
                p: Principal = Depends(auth.require("user:admin"))):
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(404, "User not found")
    _check_user_input(u.username, body.roles, body.password)
    if u.id == p.id and (body.disabled or (body.roles is not None and "admin" not in body.roles)):
        raise HTTPException(409, "You cannot disable yourself or remove your own admin role")
    changes = {}
    if body.display_name is not None:
        u.display_name = body.display_name.strip()
        changes["display_name"] = True
    if body.roles is not None:
        u.roles = sorted(set(body.roles))
        changes["roles"] = u.roles
    if body.disabled is not None:
        u.disabled = body.disabled
        changes["disabled"] = body.disabled
    if body.password:
        u.pw_hash = auth.hash_password(body.password)
        u.must_change_password = True
        u.locked_until = None
        changes["password_reset"] = True
    if body.disabled or body.password or body.roles is not None:
        auth.revoke_user_sessions(db, u.id)
    audit.record(db, p.actor, "user.updated", None, {"user": u.username, **changes})
    return _user_json(u)


@app.get("/api/official")
def official_summary(p: Principal = Depends(auth.current)):
    """Published NCRB / I4C statistics (verbatim files from data.gov.in, hash-checked)."""
    return official.summary()


@app.get("/api/official/context")
def official_context(district: str, p: Principal = Depends(auth.current)):
    ctx = official.context_for(district)
    if not ctx:
        raise HTTPException(404, "No official statistics mapped for this district")
    return ctx


@app.get("/api/stats")
def stats(training: bool = False, db: Session = Depends(db_session), p: Principal = Depends(CASE_READ)):
    cases = db.execute(select(Case).where(Case.synthetic.is_(training))).scalars().all()
    open_ = [c for c in cases if c.status != "closed"]
    rings = linker.rings(db, training=training)
    timed = [c.intake_seconds for c in cases if c.intake_seconds]
    since = utcnow() - timedelta(hours=24)
    return {
        "open_cases": len(open_), "immediate": sum(1 for c in open_ if c.priority == "IMMEDIATE"),
        "safety_alerts": sum(1 for c in open_ if (c.safety or {}).get("coercion") or (c.safety or {}).get("crisis")),
        "rings": len([r for r in rings if r["size"] >= 2]),
        "linked_cases": sum(r["size"] for r in rings),
        "reported_loss": sum(_amount(c) for c in open_),
        "median_intake_seconds": float(np.median(timed)) if timed else None,
        "last_24h": sum(1 for c in cases if c.created_at >= since),
        "workspace": "training" if training else "live",
        "training_available": db.execute(select(func.count(Case.id)).where(Case.synthetic.is_(True))).scalar() or 0,
    }


@app.get("/api/cases")
def list_cases(status: str | None = None, priority: str | None = None, training: bool = False,
               db: Session = Depends(db_session), p: Principal = Depends(CASE_READ)):
    q = select(Case).where(Case.synthetic.is_(training))
    if status:
        q = q.where(Case.status == status)
    if priority:
        q = q.where(Case.priority == priority)
    rows = db.execute(q).scalars().all()
    rows.sort(key=lambda c: (-c.priority_score, -(c.created_at.timestamp())))
    return [case_summary(c, p) for c in rows]


@app.post("/api/cases", status_code=201)
def create_case(body: CaseIn, db: Session = Depends(db_session), p: Principal = Depends(CASE_WRITE)):
    who = p.actor
    now = datetime.now()
    year = now.year
    n = (db.execute(select(func.count(Case.id))).scalar() or 0) + 1
    training = body.synthetic and p.can("link:review")  # only supervisors may create training scenarios
    case = Case(id=f"AV-{year}-{n:05d}", mode=body.mode, channel=body.channel, caller_label=body.caller_label,
                district=body.district, synthetic=training, created_by=p.username,
                language=None if body.language in (None, "", "auto") else body.language,
                call_started_at=(body.call_started_at or now).replace(tzinfo=None))
    while pipeline.get_case(db, case.id):
        n += 1
        case.id = f"AV-{year}-{n:05d}"
    db.add(case)
    db.flush()
    audit.record(db, who, "case.created", case.id, {"mode": body.mode, "channel": body.channel,
                                                    "synthetic": training})
    pipeline.process(db, case, who)
    return case_detail(db, case, p)


@app.get("/api/cases/{case_id}")
def get_case(case_id: Annotated[str, CaseId], db: Session = Depends(db_session), p: Principal = Depends(CASE_READ)):
    who = p.actor
    case = _get(db, case_id)
    if case.distress and "voice_score" not in case.distress:  # analysed by an older version: refresh once
        pipeline.process(db, case, "system")
    key = (p.username, case.id)
    if time.monotonic() - _viewed.get(key, -1e9) > 600:  # access log, at most once per officer per case per 10 min
        _viewed[key] = time.monotonic()
        audit.record(db, who, "case.viewed", case.id, {"identity_revealed": case.mode == "informant" and _can_reveal(p, case)})
    return case_detail(db, case, p)


@app.patch("/api/cases/{case_id}")
def patch_case(case_id: Annotated[str, CaseId], body: CasePatch, db: Session = Depends(db_session), p: Principal = Depends(CASE_WRITE)):
    who = p.actor
    case = _get(db, case_id)
    changes = body.model_dump(exclude_none=True)
    for k, v in changes.items():
        setattr(case, k, v)
    audit.record(db, who, "case.updated", case.id, changes)
    pipeline.process(db, case, who)
    return case_detail(db, case, p)


@app.post("/api/cases/{case_id}/segments")
def add_segments(case_id: Annotated[str, CaseId], body: SegmentsIn, db: Session = Depends(db_session), p: Principal = Depends(CASE_WRITE)):
    """Add transcript lines (operator notes, a typed transcript, or seed data)."""
    who = p.actor
    case = _get(db, case_id)
    pipeline.add_segments(db, case, [s.model_dump() for s in body.segments])
    audit.record(db, who, "transcript.added", case.id, {"segments": len(body.segments), "source": "text"})
    db.refresh(case)
    pipeline.process(db, case, who)
    return case_detail(db, case, p)


@app.patch("/api/cases/{case_id}/segments/{seg_id}")
def label_segment(case_id: Annotated[str, CaseId], seg_id: Annotated[str, ObjId], body: SpeakerPatch, db: Session = Depends(db_session),
                  p: Principal = Depends(CASE_WRITE)):
    who = p.actor
    case = _get(db, case_id)
    seg = db.get(Segment, seg_id)
    if not seg or seg.case_id != case.id:
        raise HTTPException(404, "Segment not found")
    seg.speaker = body.speaker
    audit.record(db, who, "speaker.labelled", case.id, {"segment_id": seg_id, "speaker": body.speaker})
    pipeline.process(db, case, who)
    return case_detail(db, case, p)


@app.post("/api/cases/{case_id}/recordings")
async def upload_recording(case_id: Annotated[str, CaseId], file: UploadFile = File(...), kind: str = Form("call"),
                           db: Session = Depends(db_session), p: Principal = Depends(CASE_WRITE)):
    who = p.actor
    if kind not in ("call", "suspect_audio"):
        raise HTTPException(422, "kind must be 'call' or 'suspect_audio'")
    case = _get(db, case_id)
    if file.content_type and not file.content_type.startswith(AUDIO_TYPES):
        raise HTTPException(415, "Upload an audio file")
    data = await _read_limited(file, MAX_UPLOAD_BYTES)
    if not data:
        raise HTTPException(422, "Empty file")
    st = M.asr.status()
    if st.state in ("disabled", "error"):
        raise HTTPException(503, f"Speech-to-text model is not available on this server: {st.error}. "
                                 "Install requirements-ml.txt (see README).")
    offset = max((s.end for s in case.segments), default=0.0)
    try:
        x = await asyncio.to_thread(A.decode, data)
    except Exception:
        raise HTTPException(422, "Could not decode this file as audio") from None
    model = M.best_asr()
    res = await asyncio.to_thread(pipeline.transcribe_and_analyse, x, case.mode, kind, offset, True, True, "", model,
                                  case.language)
    rec = pipeline.ingest_audio(db, case, data, _safe_filename(file.filename), file.content_type, kind, who, offset,
                                precomputed=res)
    refine_later = model is M.asr and M.asr_final.enabled
    db.refresh(case)
    pipeline.process(db, case, who)
    if refine_later:
        db.commit()
        pipeline.refine_async(case.id, rec.id, who)
    return case_detail(db, case, p)


@app.post("/api/recordings/{rec_id}/refine")
def refine(rec_id: Annotated[str, ObjId], db: Session = Depends(db_session), p: Principal = Depends(CASE_WRITE)):
    """Re-transcribe a recording with the accurate model (runs in the background)."""
    who = p.actor
    rec = db.get(Recording, rec_id)
    if not rec:
        raise HTTPException(404, "Recording not found")
    pipeline.refine_async(rec.case_id, rec.id, who)
    return {"ok": True}


@app.get("/api/recordings/{rec_id}/audio")
def recording_audio(rec_id: Annotated[str, ObjId], db: Session = Depends(db_session), p: Principal = Depends(CASE_READ)):
    who = p.actor
    rec = db.get(Recording, rec_id)
    if not rec:
        raise HTTPException(404, "Recording not found")
    if not _can_reveal(p, db.get(Case, rec.case_id)):
        raise HTTPException(403, "Recording of a protected informant: a supervisor must open it")
    try:
        data = storage.load(rec.stored_path)
    except storage.EvidenceError:
        audit.record(db, who, "evidence.integrity_failed", rec.case_id, {"recording_id": rec.id})
        db.commit()  # keep the security event even though the request fails
        raise HTTPException(409, "Integrity check failed: stored evidence could not be authenticated") from None
    if audit.sha256_bytes(data) != rec.sha256:
        audit.record(db, who, "evidence.integrity_failed", rec.case_id, {"recording_id": rec.id})
        db.commit()
        raise HTTPException(409, "Integrity check failed: stored evidence does not match its recorded hash")
    audit.record(db, who, "evidence.accessed", rec.case_id, {"recording_id": rec.id, "sha256": rec.sha256})
    return Response(content=data, media_type=rec.mime or "audio/wav",
                    headers={"X-Evidence-SHA256": rec.sha256, "Cache-Control": "no-store"})


@app.patch("/api/cases/{case_id}/fields/{name}")
def act_on_field(case_id: Annotated[str, CaseId], name: Annotated[str, KeyName], body: FieldAction, db: Session = Depends(db_session),
                 p: Principal = Depends(CASE_WRITE)):
    who = p.actor
    case = _get(db, case_id)
    row = next((f for f in case.fields if f.name == name), None)
    if body.action == "correct" and row is None:
        row = FieldRow(case_id=case.id, name=name, value="", display="", confidence=1.0, evidence=[],
                       model_value=None)
        db.add(row)
    if row is None:
        raise HTTPException(404, "Field not found")
    before = {"value": row.value, "status": row.status}
    if body.action == "verify":
        row.status = "verified"
    elif body.action == "reported":
        row.status = "reported"
    elif body.action == "correct":
        if body.value is None:
            raise HTTPException(422, "value is required to correct a field")
        row.value = body.value
        row.display = body.display or body.value
        row.status = "corrected"
        row.confidence = 1.0
    elif body.action == "reject":
        db.delete(row)
    row.updated_by = who
    audit.record(db, who, f"field.{body.action}", case.id, {"field": name, "before": before,
                                                             "after": {"value": body.value}})
    db.flush()
    db.refresh(case)
    pipeline.process(db, case, who)
    return case_detail(db, case, p)


@app.post("/api/cases/{case_id}/copilot/{key}/asked")
def mark_asked(case_id: Annotated[str, CaseId], key: Annotated[str, KeyName], db: Session = Depends(db_session), p: Principal = Depends(CASE_WRITE)):
    who = p.actor
    case = _get(db, case_id)
    case.asked = sorted(set(case.asked or []) | {key})
    pipeline.process(db, case, who)
    return case_detail(db, case, p)


@app.post("/api/cases/{case_id}/complete")
def complete_intake(case_id: Annotated[str, CaseId], db: Session = Depends(db_session), p: Principal = Depends(CASE_WRITE)):
    who = p.actor
    case = _get(db, case_id)
    case.status = "review"
    case.intake_seconds = round((utcnow() - case.created_at).total_seconds(), 1)
    audit.record(db, who, "intake.completed", case.id, {"intake_seconds": case.intake_seconds})
    return case_detail(db, case, p)


@app.get("/api/cases/{case_id}/readback")
def readback(case_id: Annotated[str, CaseId], lang: str = Query("hi", pattern="^(hi|en)$"), db: Session = Depends(db_session),
             p: Principal = Depends(CASE_READ)):
    who = p.actor
    case = _get(db, case_id)
    text = pipeline.readback(case, lang)
    audit.record(db, who, "readback.generated", case.id, {"lang": lang})
    return {"lang": lang, "text": text}


@app.get("/api/cases/{case_id}/report")
def report(case_id: Annotated[str, CaseId], db: Session = Depends(db_session), p: Principal = Depends(CASE_READ)):
    """Editable complaint draft; every field points to its evidence. Not a legal determination."""
    who = p.actor
    case = _get(db, case_id)
    detail = case_detail(db, case, p)
    head = audit.verify(db)
    audit.record(db, who, "report.generated", case.id)
    return {
        "title": f"Complaint draft · {case.id}",
        "classification": "CONFIDENTIAL · contains personal data of a complainant. Share only with authorised officers.",
        "generated_by": p.actor, "generated_at": utcnow().isoformat() + "Z",
        "disclaimer": "Draft for officer review. AI outputs are indicative and must be verified; "
                      "this document does not determine guilt or credibility.",
        "case": {k: detail[k] for k in ("id", "mode", "channel", "caller_label", "district", "created_at",
                                        "priority", "playbook")},
        "fields": detail["fields"], "triage": detail["triage"], "tactics": detail["tactics"],
        "links": detail["links"], "recordings": detail["recordings"],
        "integrity": {"audit_chain_ok": head["ok"], "audit_head": head.get("head"),
                      "pipeline": pipeline.model_version()},
    }


class EraseIn(BaseModel):
    reason: str = PField(min_length=5, max_length=200)


@app.post("/api/cases/{case_id}/erase")
def erase(body: EraseIn, case_id: Annotated[str, CaseId], db: Session = Depends(db_session),
          p: Principal = Depends(auth.require("link:review"))):
    """Supervisor-only: permanently erase a case and its encrypted evidence (e.g. retention, wrong intake)."""
    retention.erase_case(db, _get(db, case_id), p.actor, body.reason)
    return {"ok": True}


@app.get("/api/graph")
def graph(training: bool = False, db: Session = Depends(db_session), p: Principal = Depends(CASE_READ)):
    pool = set(db.execute(select(Case.id).where(Case.synthetic.is_(training))).scalars())
    links = [l for l in db.execute(select(Link).where(Link.status != "rejected")).scalars()
             if l.case_a in pool and l.case_b in pool]
    ids = {l.case_a for l in links} | {l.case_b for l in links}
    cases = db.execute(select(Case).where(Case.id.in_(ids))).scalars().all() if ids else []
    return {"nodes": [case_summary(c, p) for c in cases],
            "edges": [{"id": l.id, "source": l.case_a, "target": l.case_b, "kind": l.kind, "score": l.score,
                       "status": l.status, "reason": l.detail.get("reason")} for l in links],
            "rings": linker.rings(db, training=training)}


@app.patch("/api/links/{link_id}")
def review_link(link_id: Annotated[int, Path(ge=1)], body: LinkPatch, db: Session = Depends(db_session), p: Principal = Depends(auth.require("link:review"))):
    who = p.actor
    l = db.get(Link, link_id)
    if not l:
        raise HTTPException(404, "Link not found")
    l.status = body.status
    audit.record(db, who, f"link.{body.status}", l.case_a, {"link_id": l.id, "case_b": l.case_b, "kind": l.kind})
    return {"id": l.id, "status": l.status}


@app.get("/api/audit")
def audit_log(case_id: str | None = None, limit: int = 200, db: Session = Depends(db_session),
              p: Principal = Depends(auth.require("audit:read"))):
    q = select(AuditEvent).order_by(AuditEvent.id.desc()).limit(min(limit, 1000))
    if case_id:
        q = q.where(AuditEvent.case_id == case_id)
    return [{"id": e.id, "ts": e.ts.isoformat() + "Z", "actor": e.actor, "action": e.action, "case_id": e.case_id,
             "payload": e.payload, "hash": e.hash, "prev_hash": e.prev_hash, "model_version": e.model_version}
            for e in db.execute(q).scalars()]


@app.get("/api/audit/verify")
def audit_verify(db: Session = Depends(db_session), p: Principal = Depends(auth.require("audit:read"))):
    return audit.verify(db)


@app.get("/api/radar")
def radar(days: int = 7, training: bool = False, db: Session = Depends(db_session), p: Principal = Depends(CASE_READ)):
    """Campaign radar: which scam types are rising, where. Counts complaints, not verified crimes."""
    now = utcnow()
    since = now - timedelta(days=days)
    cases = [c for c in db.execute(select(Case).where(Case.created_at >= since, Case.synthetic.is_(training))).scalars() if c.mode != "suspect_recording"]
    out = []
    for key in {c.playbook_key or "other" for c in cases}:
        group = [c for c in cases if (c.playbook_key or "other") == key]
        last24 = [c for c in group if c.created_at >= now - timedelta(hours=24)]
        prior_daily = (len(group) - len(last24)) / max(days - 1, 1)
        pb = playbooks.PLAYBOOK_BY_KEY.get(key)
        out.append({
            "key": key, "label": pb.label if pb else "Type unclear", "last_24h": len(last24), "last_period": len(group),
            "prior_daily_avg": round(prior_daily, 2), "rising": len(last24) >= 2 and len(last24) > 2 * prior_daily,
            "districts": sorted({c.district for c in group if c.district}),
            "reported_loss": sum(_amount(c) for c in group),
        })
    out.sort(key=lambda r: (-r["last_24h"], -r["last_period"]))
    return {"days": days, "groups": out}


@app.get("/api/playbooks")
def list_playbooks(p: Principal = Depends(auth.current)):
    return [{"key": pb.key, "label": pb.label, "required": list(pb.required), "advice": pb.advice}
            for pb in [*playbooks.PLAYBOOKS, playbooks.GENERIC]]


# --- live streaming ---------------------------------------------------------------------------

@app.websocket("/ws/cases/{case_id}/live")
async def live(ws: WebSocket, case_id: str):
    """Browser sends 16 kHz mono PCM16 frames; server transcribes ~5 s chunks and pushes case updates.

    Authenticated by the session cookie; the Origin header must be an allowed frontend origin
    (blocks cross-site WebSocket hijacking). Frames and session length are capped."""
    if ws.headers.get("origin") not in settings.cors_origins or not re.match(r"^AV-\d{4}-[A-Z0-9]{4,6}$", case_id):
        log.warning("live %s refused: origin %r not allowed (open the app at one of %s)", case_id,
                    ws.headers.get("origin"), ", ".join(settings.cors_origins))
        await ws.close(code=4403)
        return
    with SessionLocal() as dba:
        p = auth.principal_from_token(dba, ws.cookies.get(auth.COOKIE))
        dba.commit()
    if p is None or not p.can("case:write") or p.must_change_password:
        log.warning("live %s refused: %s", case_id, "no valid session cookie" if p is None else "role cannot take calls")
        await ws.close(code=4401)
        return
    if not auth.limiter.allow(f"ws:{p.id}", 20, 60):
        await ws.close(code=4429)
        return
    await ws.accept()
    who = p.actor
    st = M.asr.status()
    if st.state in ("disabled", "error"):
        await ws.send_json({"type": "error", "message": f"Speech-to-text model not available: {st.error}"})
        await ws.close()
        return
    if st.state != "ready":
        M.asr.start_loading()
        await ws.send_json({"type": "status", "stage": "loading", "message": "Speech-to-text model is still loading; "
                            "audio is buffered and will be transcribed as soon as it is ready."})
    with SessionLocal() as db0:
        c0 = pipeline.get_case(db0, case_id)
        if not c0:
            await ws.send_json({"type": "error", "message": "Case not found"})
            await ws.close()
            return
        start_offset = max((s.end for s in c0.segments), default=0.0)
        mode = c0.mode
        call_lang = {"v": c0.language}  # set at intake, or detected on the first chunk and then kept for the call
    if st.state == "ready":
        await ws.send_json({"type": "status", "message": "Listening", "stage": "listening"})

    # Receiving never waits for the models: audio goes into two queues, served by two workers.
    full = bytearray()          # the whole call, saved as evidence at the end
    buf = bytearray()           # audio not yet sent for transcription
    buf_start = start_offset    # case-timeline time of buf[0]
    emo_ring = bytearray()      # last WINDOW_S of audio for the emotion track
    emo_since = 0               # bytes received since the last emotion window
    asr_q: asyncio.Queue = asyncio.Queue()
    emo_q: asyncio.Queue = asyncio.Queue(maxsize=2)
    chunker = live_ser.Chunker(target_s=settings.live_chunk_seconds)
    tracker = live_ser.LiveEmotionTracker(M.emotion.predict if mode != "suspect_recording" else None,
                                          role=pipeline.ROLE.get(mode, "victim"))
    pending_windows: list[dict] = []
    voice_busy = {"on": False, "next_s": 20.0}

    async def voice_check(pcm: bytes, t_end: float) -> None:
        try:
            va = await asyncio.to_thread(pipeline.check_voice, A.pcm16_to_float(pcm), mode)
            await send({"type": "voice_auth", "t": round(t_end, 1), **{k: va.get(k) for k in
                        ("verdict", "score", "model_score", "cue_score", "reasons", "abstain", "advice")}})
        except Exception:  # pragma: no cover
            log.exception("live voice check failed")
        finally:
            voice_busy["on"] = False
    prev_text: list[str] = []
    stats = {"chunks": 0, "empty": 0}
    send_lock = asyncio.Lock()

    async def send(msg: dict) -> None:
        async with send_lock:
            try:
                await ws.send_json(_finite(msg))
            except Exception:
                log.exception("live %s: could not send %s message", case_id, msg.get("type"))

    async def asr_worker() -> None:
        while True:
            item = await asr_q.get()
            try:
                if item is None:
                    return
                t0, pcm, queued = item
                x = A.pcm16_to_float(pcm)
                rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
                started = time.monotonic()
                await send({"type": "status", "stage": "transcribing", "message": "Transcribing…",
                            "backlog": asr_q.qsize()})
                res = (await asyncio.to_thread(pipeline.transcribe_and_analyse, x, mode, "call", t0, True, False,
                                               " ".join(prev_text[-3:]), None, call_lang["v"])
                       ) if len(x) >= A.SR // 2 else {"segments": [], "language": None}
                took = time.monotonic() - started
                stats["chunks"] += 1
                stats["empty"] += not res["segments"]
                log.info("live %s: chunk %.1fs rms=%.4f -> %d segments in %.1fs (%s)", case_id, len(x) / A.SR, rms,
                         len(res["segments"]), took, getattr(M.asr, "last_stats", {}))
                prev_text.extend(sg.get("text_native") or sg["text"] for sg in res["segments"])
                with SessionLocal() as db:
                    case = pipeline.get_case(db, case_id)
                    if res["segments"]:
                        pipeline.add_segments(db, case, res["segments"])
                    case.language = case.language or res.get("language")
                    if res["segments"]:
                        call_lang["v"] = case.language
                    if pending_windows:
                        case.windows = list(case.windows or []) + pending_windows
                        pending_windows.clear()
                    db.flush()
                    db.refresh(case)
                    pipeline.process(db, case, who)
                    db.commit()
                    detail = case_detail(db, case, p)
                hint = None
                if not res["segments"]:
                    hint = ("No speech heard: check that the right microphone is selected and not muted."
                            if rms < 0.004 else "Speech was too unclear to transcribe; ask the caller to repeat.")
                lag_s = time.monotonic() - queued  # end of the spoken chunk -> transcript and case file on screen
                if res["segments"]:
                    live_ser.LATENCY.add("transcript_s", lag_s)
                await send({"type": "update", "case": detail, "final": False, "hint": hint,
                            "latency_s": round(took, 1), "lag_s": round(lag_s, 1), "backlog": asr_q.qsize()})
            except Exception:  # pragma: no cover - keep the call alive
                log.exception("live transcription chunk failed")
                await send({"type": "status", "stage": "error", "message": "One segment could not be transcribed."})
            finally:
                asr_q.task_done()

    async def emo_worker() -> None:
        while True:
            item = await emo_q.get()
            if item is None:
                return
            t0, pcm, queued = item
            try:
                rec = await asyncio.to_thread(tracker.analyse, A.pcm16_to_float(pcm), t0)
                lag_ms = round(1000 * (time.monotonic() - queued))  # last audio sample in -> signal out
                if not rec["silent"]:
                    live_ser.LATENCY.add("emotion_ms", lag_ms)
                if not rec["silent"]:
                    pending_windows.append({k: rec[k] for k in ("t_start", "t_end", "emotions", "acoustic", "silent",
                                                                "label", "p")})
                await send({"type": "emotion", "t": rec["t_end"], "silent": rec["silent"], "label": rec["label"],
                            "p": rec["p"], "arousal": rec["arousal"], "indicators": rec["indicators"],
                            "model": M.emotion.state if mode != "suspect_recording" else "off",
                            "latency_ms": lag_ms})
            except Exception:  # pragma: no cover
                log.exception("live emotion window failed")

    workers = [asyncio.create_task(asr_worker()), asyncio.create_task(emo_worker())]
    max_bytes = int(A.SR * 2 * 60 * MAX_LIVE_MINUTES)
    started = time.monotonic()
    stopped = False

    def cut(final: bool = False) -> None:
        nonlocal buf, buf_start
        if buf and (final or len(buf) >= A.SR * 2 * 0.5):
            asr_q.put_nowait((buf_start, bytes(buf), time.monotonic()))
            buf_start += len(buf) / (A.SR * 2)
            buf = bytearray()

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                data = msg["bytes"]
                if len(data) > 256 * 1024 or len(full) > max_bytes or time.monotonic() - started > MAX_LIVE_MINUTES * 60:
                    await send({"type": "error", "message": "Live session limit reached; saving the call."})
                    stopped = True
                    break
                if len(data) % 2:
                    data = data[:-1]
                full.extend(data)
                buf.extend(data)
                chunker.observe(data)
                emo_ring.extend(data)
                win_bytes = int(live_ser.WINDOW_S * A.SR) * 2
                if len(emo_ring) > win_bytes:
                    del emo_ring[:len(emo_ring) - win_bytes]
                emo_since += len(data)
                if emo_since >= int(live_ser.HOP_S * A.SR) * 2 and len(emo_ring) >= win_bytes * 0.8 and not emo_q.full():
                    emo_since = 0
                    t_end = start_offset + len(full) / (A.SR * 2)
                    emo_q.put_nowait((t_end - len(emo_ring) / (A.SR * 2), bytes(emo_ring), time.monotonic()))
                secs = len(full) / (A.SR * 2)
                # synthetic-voice check every ~20 s, only when transcription is not behind (CPU priority)
                if secs >= voice_busy["next_s"] and not voice_busy["on"] and asr_q.qsize() == 0:
                    voice_busy.update(on=True, next_s=secs + 20.0)
                    asyncio.create_task(voice_check(bytes(full[-int(A.SR * 12) * 2:]), start_offset + secs))
                tail = bytes(buf[-int(A.SR * 0.5) * 2:])
                if chunker.should_cut(len(buf) / (A.SR * 2), tail):
                    cut()
            elif msg.get("text"):
                if len(msg["text"]) <= 1024 and '"stop"' in msg["text"]:
                    stopped = True
                    break
    except WebSocketDisconnect:
        pass
    except Exception:  # pragma: no cover
        log.exception("live session failed")
        await send({"type": "error", "message": "Live processing failed; see the server log."})

    # Finish: transcribe what is left, then save the whole call as evidence (also if the browser vanished).
    cut(final=True)
    if stopped:
        await send({"type": "status", "stage": "saving", "message": "Finishing the transcript and saving the call…",
                    "backlog": asr_q.qsize()})
    await asr_q.join()
    await emo_q.put(None)
    await asr_q.put(None)
    await asyncio.gather(*workers, return_exceptions=True)
    detail = None
    if full:
        full_x = A.pcm16_to_float(bytes(full))
        va = await asyncio.to_thread(pipeline.check_voice, full_x, mode)
        with SessionLocal() as db:
            case = pipeline.get_case(db, case_id)
            if pending_windows:
                case.windows = list(case.windows or []) + pending_windows
            wav = A.to_wav_bytes(full_x)
            rec = pipeline.ingest_audio(db, case, wav, f"live-{case.id}.wav", "audio/wav", "call", who,
                                        start_offset, precomputed={"voice_auth": va})
            for sg in case.segments:  # tie this session's lines to the saved audio for playback
                if sg.recording_id is None and sg.start >= start_offset - 0.01:
                    sg.recording_id = rec.id
            if M.asr_final.enabled:
                case.processing = f"Refining transcript with {M.asr_final.model_id}…"
            db.flush()
            db.refresh(case)
            pipeline.process(db, case, who)
            db.commit()
            detail = case_detail(db, case, p)
            rec_id = rec.id
        log.info("live %s: saved %.1fs, %d chunks (%d empty)", case_id, len(full) / (A.SR * 2), stats["chunks"], stats["empty"])
        if M.asr_final.enabled:
            pipeline.refine_async(case_id, rec_id, who)
    await send({"type": "update", "case": detail, "final": True} if detail else {"type": "status", "stage": "done", "message": "No audio was received."})
    try:
        await ws.close()
    except Exception:
        pass
