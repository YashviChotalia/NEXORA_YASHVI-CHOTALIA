"""HTTP routes for helpline hand-offs (see handoff.py)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field as PField
from sqlalchemy.orm import Session

from . import audit, auth, handoff, pipeline
from .auth import Principal
from .models import Handoff

router = APIRouter()
db_session = auth.db_dep
CaseId = Path(pattern=r"^AV-\d{4}-[A-Z0-9]{4,6}$")
HandoffId = Path(pattern=r"^HO-[a-f0-9]{10}$")


class CancelIn(BaseModel):
    reason: str = PField(default="", max_length=200)


class StartIn(BaseModel):
    helpline: str = PField(pattern="^(telemanas|emergency|women|child|cyber1930)$")
    reason: str = PField(default="", max_length=200)


def _case(db: Session, case_id: str):
    case = pipeline.get_case(db, case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    return case


def _handoff(db: Session, case_id: str, handoff_id: str) -> Handoff:
    h = db.get(Handoff, handoff_id)
    if h is None or h.case_id != case_id:
        raise HTTPException(404, "Hand-off not found")
    return h


def for_case(db: Session, case_id: str) -> list[dict]:
    rows = db.query(Handoff).filter(Handoff.case_id == case_id).order_by(Handoff.created_at).all()
    return [handoff.as_json(h) for h in rows]


@router.get("/api/handoffs/helplines")
def helplines(p: Principal = Depends(auth.current)):
    return {"mode": handoff.MODE, "countdown_s": handoff.COUNTDOWN_S,
            "helplines": [{"key": h.key, "name": h.name, "number": h.number, "auto": h.auto, "why": h.why}
                          for h in handoff.HELPLINES.values()]}


@router.get("/api/cases/{case_id}/handoffs")
def list_handoffs(case_id: str = CaseId, db: Session = Depends(db_session),
                  p: Principal = Depends(auth.require("case:read"))):
    _case(db, case_id)
    return for_case(db, case_id)


@router.post("/api/cases/{case_id}/handoffs")
def start_handoff(body: StartIn, case_id: str = CaseId, db: Session = Depends(db_session),
                  p: Principal = Depends(auth.require("case:write"))):
    """The officer transfers the call now (any helpline), or accepts a suggestion."""
    import uuid

    from .models import utcnow
    case = _case(db, case_id)
    hl = handoff.HELPLINES[body.helpline]
    h = (db.query(Handoff).filter(Handoff.case_id == case_id, Handoff.helpline == hl.key,
                                  Handoff.status.in_(("suggested", "pending"))).first())
    if h is None:
        h = Handoff(id="HO-" + uuid.uuid4().hex[:10], case_id=case.id, helpline=hl.key, number=hl.number,
                    reason=body.reason or "Transferred by the officer", status="pending", trigger="officer",
                    created_at=utcnow(), call_ref=case.id)
        db.add(h)
        db.flush()
    handoff.connect(db, h, actor=p.actor)
    db.commit()
    return handoff.as_json(h)


@router.post("/api/cases/{case_id}/handoffs/{handoff_id}/connect")
def connect_now(case_id: str = CaseId, handoff_id: str = HandoffId, db: Session = Depends(db_session),
                p: Principal = Depends(auth.require("case:write"))):
    h = _handoff(db, case_id, handoff_id)
    if h.status not in ("pending", "suggested", "failed"):
        raise HTTPException(409, f"Hand-off is already {h.status}")
    handoff.connect(db, h, actor=p.actor)
    db.commit()
    return handoff.as_json(h)


@router.post("/api/cases/{case_id}/handoffs/{handoff_id}/cancel")
def cancel_handoff(body: CancelIn, case_id: str = CaseId, handoff_id: str = HandoffId,
                   db: Session = Depends(db_session), p: Principal = Depends(auth.require("case:write"))):
    h = _handoff(db, case_id, handoff_id)
    if h.status in handoff.TERMINAL:
        raise HTTPException(409, f"Hand-off is already {h.status}")
    handoff.cancel(db, h, actor=p.actor, reason=body.reason)
    db.commit()
    return handoff.as_json(h)


__all__ = ["router", "for_case", "audit"]
