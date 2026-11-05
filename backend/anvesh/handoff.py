"""Warm hand-off of a live call to the right helpline, without the caller having to do anything.

When a live victim or informant call shows a human-safety signal, Anvesh starts a transfer itself:

  crisis        suicide / self-harm / strongly self-demeaning words   -> Tele-MANAS 14416   (automatic)
  danger        a threat to the caller's life or safety right now      -> Emergency 112      (automatic)
  women         a woman facing sextortion, stalking or harassment     -> Women helpline 181 (officer confirms)
  child         the person at risk is a child                         -> Childline 1098     (officer confirms)
  financial     money lost within the golden hour                     -> Cyber helpline 1930 (officer confirms)

"Automatic" means: the officer sees "Connecting to Tele-MANAS in 10 s" with a Cancel button, and if nobody
cancels, the call is connected. It is a WARM, three-way transfer (conference): the officer stays on the line
until the counsellor has taken over, so a person in crisis is never dropped into a ringing tone. Every step is
written to the tamper-evident audit log.

Anvesh does not run the phone line itself. `Telephony` sends the transfer to the cyber cell's phone system:
  ANVESH_TELEPHONY=demo      (default) the whole flow runs and is logged, no real call is placed
  ANVESH_TELEPHONY=webhook   POST the transfer to ANVESH_TELEPHONY_URL (signed with ANVESH_TELEPHONY_SECRET);
                             a small bridge on the PBX / cloud contact centre (Asterisk, Exotel, Twilio ...) dials
                             the helpline into the call and answers {"status": "connected" | "failed", "ref": ...}
  ANVESH_TELEPHONY=off       suggestions only, never a transfer
Words-based signals are indicative; the countdown exists so a person can stop a false alarm, and a real one is
never delayed by waiting for a click.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import timedelta

log = logging.getLogger("anvesh.handoff")


@dataclass(frozen=True)
class Helpline:
    key: str
    name: str
    number: str
    auto: bool  # transfer without a click (after the countdown) during a live call
    why: str


HELPLINES = {
    "telemanas": Helpline("telemanas", "Tele-MANAS (mental health, 24x7)", "14416", True,
                          "Words about suicide, self-harm or worthlessness"),
    "emergency": Helpline("emergency", "Emergency (police, ambulance)", "112", True,
                          "A threat to the caller's life or safety right now"),
    "women": Helpline("women", "Women helpline", "181", False, "A woman facing sextortion, stalking or harassment"),
    "child": Helpline("child", "Childline", "1098", False, "The person at risk is a child"),
    "cyber1930": Helpline("cyber1930", "National cyber crime helpline", "1930", False,
                          "Money lost within the golden hour: freeze it through 1930"),
}

COUNTDOWN_S = float(os.getenv("ANVESH_HANDOFF_COUNTDOWN_S", "10"))
MODE = os.getenv("ANVESH_TELEPHONY", "demo").lower()
URL = os.getenv("ANVESH_TELEPHONY_URL", "")
SECRET = os.getenv("ANVESH_TELEPHONY_SECRET", "")
TERMINAL = ("connected", "cancelled", "failed", "dismissed")
LIVE_CASES: set[str] = set()  # cases with a live call in progress (set by the live WebSocket)


# --------------------------------------------------------------------------- what to offer

def needed(case, live: bool) -> list[tuple[Helpline, str, list[dict]]]:
    """Helplines this case calls for, with the reason and the quoted words, from the case's safety signals."""
    if case.mode not in ("victim", "informant"):
        return []
    s = case.safety or {}
    out = []
    if s.get("crisis"):
        out.append((HELPLINES["telemanas"], "Crisis words: " + _quote(s.get("crisis_hits")), s.get("crisis_hits") or []))
    if s.get("danger"):
        out.append((HELPLINES["emergency"], "Danger to life: " + _quote(s.get("danger_hits")), s.get("danger_hits") or []))
    if s.get("women"):
        out.append((HELPLINES["women"], "Women's safety: " + _quote(s.get("women_hits")), s.get("women_hits") or []))
    if s.get("child"):
        out.append((HELPLINES["child"], "A child may be at risk: " + _quote(s.get("child_hits")), s.get("child_hits") or []))
    tri = case.triage or {}
    if live and tri.get("priority") == "IMMEDIATE" and any("min after the incident" in r for r in tri.get("reasons", [])):
        out.append((HELPLINES["cyber1930"], "Money lost minutes ago: report on 1930 to freeze it", []))
    return out


def _quote(hits) -> str:
    qs = []
    for h in hits or []:
        q = (h.get("quote") or h.get("phrase") or "").strip()
        if q and q not in qs:
            qs.append(q)
    return ", ".join(f"“{q}”" for q in qs[:3]) or "detected in the caller's words"


# --------------------------------------------------------------------------- state changes

def on_update(db, case, actor: str, live: bool, call_ref: str | None = None) -> list:
    """Called after every case update. Starts automatic transfers (live calls only) and records suggestions.
    One hand-off per helpline per case; returns the hand-offs that were just created."""
    from .models import Handoff, utcnow
    from . import audit
    existing = {h.helpline for h in db.query(Handoff).filter(Handoff.case_id == case.id)}
    created = []
    for hl, reason, hits in needed(case, live):
        if hl.key in existing:
            continue
        auto = hl.auto and live and MODE != "off"
        now = utcnow()
        h = Handoff(id="HO-" + uuid.uuid4().hex[:10], case_id=case.id, helpline=hl.key, number=hl.number,
                    reason=reason, status="pending" if auto else "suggested", trigger="auto" if auto else "suggest",
                    created_at=now, execute_at=now + timedelta(seconds=COUNTDOWN_S) if auto else None,
                    call_ref=call_ref or case.id, evidence=[{k: x.get(k) for k in ("segment_id", "t_start", "t_end")}
                                                             for x in hits][:5])
        db.add(h)
        db.flush()
        audit.record(db, actor, "handoff.started" if auto else "handoff.suggested", case.id,
                     {"helpline": hl.key, "number": hl.number, "countdown_s": COUNTDOWN_S if auto else None,
                      "mode": MODE})
        created.append(h)
        if auto:
            schedule(h.id, COUNTDOWN_S)
    return created


def schedule(handoff_id: str, delay_s: float) -> None:
    t = threading.Timer(max(0.0, delay_s), _fire, args=(handoff_id,))
    t.daemon = True
    t.start()


def _fire(handoff_id: str) -> None:
    """Countdown over: connect unless an officer cancelled or already connected it."""
    from .models import Handoff, SessionLocal
    with SessionLocal() as db:
        h = db.get(Handoff, handoff_id)
        if h is None or h.status != "pending":
            return
        connect(db, h, actor="system:auto-handoff")
        db.commit()


def connect(db, h, actor: str) -> None:
    from .models import utcnow
    from . import audit
    hl = HELPLINES[h.helpline]
    h.status, h.executed_at, h.actor = "connecting", utcnow(), actor
    db.flush()
    ok, ref, detail = transfer(h.call_ref or h.case_id, hl.number, h.case_id, hl.key)
    h.status = "connected" if ok else "failed"
    h.provider_ref, h.detail = ref, detail
    audit.record(db, actor, "handoff.connected" if ok else "handoff.failed", h.case_id,
                 {"helpline": hl.key, "number": hl.number, "mode": MODE, "ref": ref, "detail": detail})


def cancel(db, h, actor: str, reason: str = "") -> None:
    from .models import utcnow
    from . import audit
    if h.status in TERMINAL:
        return
    h.status, h.executed_at, h.actor = ("dismissed" if h.status == "suggested" else "cancelled"), utcnow(), actor
    h.detail = reason[:200] or None
    audit.record(db, actor, "handoff.cancelled", h.case_id, {"helpline": h.helpline, "reason": reason[:200]})


# --------------------------------------------------------------------------- the phone system

def transfer(call_ref: str, number: str, case_id: str, helpline: str) -> tuple[bool, str | None, str]:
    """Ask the phone system for a warm (conference) transfer. Returns (ok, provider reference, detail)."""
    if MODE == "demo":
        ref = "DEMO-" + uuid.uuid4().hex[:8].upper()
        return True, ref, f"Demo mode: {number} joined the call as a three-way conference (no real call placed)."
    if MODE != "webhook" or not URL:
        return False, None, "No phone system configured (ANVESH_TELEPHONY / ANVESH_TELEPHONY_URL): dial manually."
    body = json.dumps({"event": "warm_transfer", "call_ref": call_ref, "to": number, "mode": "conference",
                       "case_id": case_id, "helpline": helpline}).encode()
    sig = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest() if SECRET else ""
    req = urllib.request.Request(URL, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "X-Anvesh-Signature": sig})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:  # noqa: S310 - URL is operator configuration
            res = json.loads(r.read() or b"{}")
        ok = str(res.get("status", "")).lower() in ("connected", "ok", "ringing", "bridged")
        return ok, res.get("ref"), str(res.get("detail") or res.get("status") or "")[:200]
    except Exception as e:  # network, timeout, bad JSON
        log.warning("transfer to %s failed: %s", number, e)
        return False, None, f"Phone system did not confirm ({type(e).__name__}): dial {number} manually."


def as_json(h) -> dict:
    hl = HELPLINES.get(h.helpline)
    return {"id": h.id, "helpline": h.helpline, "name": hl.name if hl else h.helpline, "number": h.number,
            "reason": h.reason, "status": h.status, "trigger": h.trigger, "mode": MODE,
            "created_at": h.created_at.isoformat() + "Z" if h.created_at else None,
            "execute_at": h.execute_at.isoformat() + "Z" if h.execute_at else None,
            "executed_at": h.executed_at.isoformat() + "Z" if h.executed_at else None,
            "actor": h.actor, "provider_ref": h.provider_ref, "detail": h.detail, "evidence": h.evidence or []}
