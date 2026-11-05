"""Transparent, configurable triage rules.

Financial urgency is driven by reported facts (recency, loss, missing critical
fields). Human-safety flags (crisis, coercion) can raise a case to immediate
human attention.

Voice distress is an *assistance* signal with a small, capped and visible role:
sustained high distress on a live victim / informant call raises the priority by
at most ONE level, and elevated distress only breaks ties inside a level. It never
applies to suspect recordings, never touches the financial score, is always shown
with its reason, and the officer can override it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

from .nlp.extract import Field, minutes_since

RULES = {
    "immediate_window_min": 24 * 60,  # loss reported within 24 h
    "high_window_min": 7 * 24 * 60,
    "large_loss_inr": 100_000,
    "critical_fields": ["amount_lost", "incident_time", "upi_id", "bank"],
    "safety_live_window_min": 6 * 60,  # crisis / coercion flags escalate for 6 h after the call
    "distress_raise_one_level": True,  # sustained high distress: +1 priority level (capped), live calls only
    "distress_tiebreak_points": 8,  # elevated distress: a few points, never across a level boundary
}

LEVELS = ("STANDARD", "HIGH", "IMMEDIATE")
LEVEL_FLOOR = {"STANDARD": 0, "HIGH": 30, "IMMEDIATE": 60}
EMOTION_ROLES = ("victim", "informant")

NEXT_ACTION_FIN = ("Follow the financial-fraud reporting procedure promptly: register on 1930 / NCRP and "
                   "contact the debited bank through its official channel.")


@dataclass
class Triage:
    priority: str  # IMMEDIATE | HIGH | STANDARD
    score: int
    headline: str
    reasons: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    next_action: str = ""
    human_safety: list[str] = field(default_factory=list)
    evidence_status: str = "Awaiting verification"
    financial_score: int = 0  # facts only; emotion never changes this
    emotion_adjustment: dict | None = None  # {"from", "to", "points", "reason"} when distress changed the order

    def to_dict(self) -> dict:
        return asdict(self)


FIELD_LABELS = {
    "amount_lost": "Reported loss", "incident_time": "Incident time", "upi_id": "Recipient UPI ID",
    "bank": "Debited bank", "phone_number": "Caller number", "transaction_id": "UTR / transaction ID",
    "account_number": "Account number", "ifsc": "IFSC", "payment_app": "Payment app",
    "claimed_agency": "Claimed agency", "platform": "Platform / app", "email": "Email",
}


def assess(mode: str, fields: dict[str, Field], now: datetime, coercion: bool, crisis: bool,
           distress_level: str | None, verified: set[str] | None = None,
           call_age_min: float | None = None) -> Triage:
    """`call_age_min`: minutes since the call. Safety flags escalate only while still actionable."""
    verified = verified or set()
    live = call_age_min is None or call_age_min <= RULES["safety_live_window_min"]
    reasons: list[str] = []
    safety: list[str] = []
    score = 0

    if mode == "suspect_recording":
        t = Triage("STANDARD", 20, "Investigative lead: suspected scam recording",
                   ["Recording analysed for scam tactics and identifiers"], [],
                   "Review flagged tactics and identifiers; link to open complaints before action.")
        return t

    amt = fields.get("amount_lost")
    mins = minutes_since(fields.get("incident_time"), now)
    amount = float(amt.value) if amt else 0.0

    if amt and mins is not None and mins <= RULES["immediate_window_min"]:
        score += 60
        reasons.append(f"Financial loss reported {mins} min after the incident")
    elif amt and mins is not None and mins <= RULES["high_window_min"]:
        score += 35
        reasons.append("Financial loss within the last 7 days")
    elif amt:
        score += 20
        reasons.append("Financial loss reported (incident time not yet captured)" if mins is None
                       else "Financial loss older than 7 days")
    if amount >= RULES["large_loss_inr"]:
        score += 15
        reasons.append(f"Large reported loss ({amt.display})")

    missing = [f for f in RULES["critical_fields"] if f not in fields]
    if missing and amt:
        score += 5
        reasons.append("Critical details still missing: " + ", ".join(FIELD_LABELS.get(m, m) for m in missing))

    if crisis:
        safety.append("Crisis language: hand over to a trained person now" if live
                      else "Crisis language during the call: arrange a welfare follow-up")
        score = max(score, 90 if live else score + 10)
    if coercion:
        safety.append("Possible active coercion: caller may still be under the scammer's control" if live
                      else "Coercion reported during the call: confirm the caller is safe")
        score = max(score, 85 if live else score + 10)
    if distress_level == "high" and not crisis:
        safety.append("Sustained high distress in the caller's voice (indicative)")

    score = min(score, 100)
    financial = score
    priority = _level(score, (crisis or coercion) and live)
    adjustment = None
    if mode in EMOTION_ROLES and live and distress_level in ("high", "elevated"):
        adjustment = _distress_adjustment(priority, score, distress_level)
        if adjustment:
            priority, score = adjustment["to"], score + adjustment["points"]
            reasons.append(adjustment["reason"])

    headline = "Financial fraud triage" if amt else "Incident intake"
    if crisis or coercion:
        headline = "Human safety first"
    status = ("Verified by officer" if verified and all(f in verified for f in fields)
              else "Partly verified" if verified else "Awaiting verification")
    return Triage(priority, score, headline, reasons, missing, NEXT_ACTION_FIN if amt else
                  "Capture the incident details; ask how it started.", safety, status, financial, adjustment)


def _level(score: int, safety_live: bool) -> str:
    if score >= 60 or safety_live:
        return "IMMEDIATE"
    return "HIGH" if score >= 30 else "STANDARD"


def _distress_adjustment(priority: str, score: int, distress_level: str) -> dict | None:
    """The only place emotion touches the queue order. Capped: one level up, or a tie-break inside a level."""
    i = LEVELS.index(priority)
    if distress_level == "high" and RULES["distress_raise_one_level"] and i < len(LEVELS) - 1:
        to = LEVELS[i + 1]
        points = max(0, LEVEL_FLOOR[to] - score)
        return {"from": priority, "to": to, "points": points,
                "reason": f"Raised one level ({priority.title()} to {to.title()}): sustained high distress "
                          "in the caller's voice and words (indicative; officer can override)"}
    ceiling = LEVEL_FLOOR[LEVELS[i + 1]] - 1 if i < len(LEVELS) - 1 else 100
    points = min(RULES["distress_tiebreak_points"], max(0, ceiling - score))
    if distress_level in ("high", "elevated") and points > 0:
        return {"from": priority, "to": priority, "points": points,
                "reason": f"Moved up within {priority.title()}: {distress_level} distress in the caller's voice "
                          "(indicative; tie-break only, never changes the level)"}
    return None
