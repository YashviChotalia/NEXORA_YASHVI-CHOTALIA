import time

from anvesh import handoff
from anvesh.nlp import safety
from helpers import make_client


class Seg:
    def __init__(self, text, speaker="caller"):
        self.id, self.text, self.start, self.end, self.speaker = "SG1", text, 0.0, 4.0, speaker


def test_self_harm_and_worthlessness_are_crisis_words():
    for line in ("mujhe nahi jeena ab, main bojh hoon sab par", "I want to die", "main kisi kaam ka nahi"):
        assert safety.scan([Seg(line)]).crisis_hits, line
    assert not safety.scan([Seg("meri galti thi ki maine OTP diya")]).crisis_hits  # ordinary self-blame


def test_danger_women_and_child_signals():
    s = safety.scan([Seg("woh log ghar ke bahar khade hain, jaan se maar denge")])
    assert s.danger_hits
    s = safety.scan([Seg("meri beti ki photo morph karke viral kar denge bol raha hai")])
    assert s.women_hits and s.child_hits


class C:
    def __init__(self, mode="victim", **safety_):
        self.id, self.mode, self.safety, self.triage = "AV-2026-T0001", mode, safety_, {}


def test_crisis_on_a_live_call_is_an_automatic_transfer_and_suspects_never_are():
    need = handoff.needed(C(crisis=True, crisis_hits=[{"quote": "main bojh hoon"}]), live=True)
    assert need[0][0].key == "telemanas" and need[0][0].number == "14416" and need[0][0].auto
    assert "main bojh hoon" in need[0][1]
    assert handoff.needed(C(mode="suspect_recording", crisis=True), live=True) == []
    keys = [h.key for h, _, _ in handoff.needed(C(danger=True, women=True, child=True), live=True)]
    assert keys == ["emergency", "women", "child"]


def test_demo_transfer_is_a_warm_conference():
    ok, ref, detail = handoff.transfer("AV-2026-T0001", "14416", "AV-2026-T0001", "telemanas")
    assert ok and ref.startswith("DEMO-") and "three-way" in detail


def _crisis_case(c):
    r = c.post("/api/cases", json={"mode": "victim", "channel": "helpline", "caller_label": "Test"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def say(c, cid, *lines):
    segs = [{"start": i * 5.0, "end": i * 5.0 + 5, "text": t, "speaker": "caller"} for i, t in enumerate(lines)]
    r = c.post(f"/api/cases/{cid}/segments", json={"segments": segs})
    assert r.status_code == 200, r.text
    return r.json()


def test_live_crisis_starts_countdown_then_connects(monkeypatch):
    c = make_client()
    cid = _crisis_case(c)
    monkeypatch.setattr(handoff, "COUNTDOWN_S", 0.3)
    handoff.LIVE_CASES.add(cid)
    try:
        d = say(c, cid, "sab khatam ho gaya, ab jeene ka mann nahi hai")
    finally:
        handoff.LIVE_CASES.discard(cid)
    ho = [h for h in d["handoffs"] if h["helpline"] == "telemanas"]
    assert ho and ho[0]["status"] == "pending" and ho[0]["execute_at"]
    time.sleep(1.0)
    after = c.get(f"/api/cases/{cid}/handoffs").json()
    assert [h["status"] for h in after if h["helpline"] == "telemanas"] == ["connected"]
    audit = c.get("/api/audit?limit=50")
    if audit.status_code == 200:  # operators may not read the audit log; supervisors can
        assert any(e["action"] == "handoff.connected" for e in audit.json())


def test_officer_can_cancel_during_the_countdown(monkeypatch):
    c = make_client()
    cid = _crisis_case(c)
    monkeypatch.setattr(handoff, "COUNTDOWN_S", 0.5)
    handoff.LIVE_CASES.add(cid)
    try:
        d = say(c, cid, "main kisi kaam ka nahi, sab par bojh hoon")
    finally:
        handoff.LIVE_CASES.discard(cid)
    hid = next(h["id"] for h in d["handoffs"] if h["helpline"] == "telemanas")
    r = c.post(f"/api/cases/{cid}/handoffs/{hid}/cancel", json={"reason": "caller was quoting a film"})
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    time.sleep(0.9)
    assert c.get(f"/api/cases/{cid}/handoffs").json()[0]["status"] == "cancelled"  # the timer did not override it


def test_after_the_call_crisis_is_only_a_suggestion():
    c = make_client()
    cid = _crisis_case(c)
    d = say(c, cid, "ab jeene ka mann nahi")
    ho = [h for h in d["handoffs"] if h["helpline"] == "telemanas"]
    assert ho and ho[0]["status"] == "suggested"
    r = c.post(f"/api/cases/{cid}/handoffs/{ho[0]['id']}/connect")
    assert r.status_code == 200 and r.json()["status"] == "connected"
