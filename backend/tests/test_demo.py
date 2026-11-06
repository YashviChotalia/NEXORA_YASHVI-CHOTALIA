"""Guided demo calls run the real analysis on scripted words and land in the Training workspace."""

import time

from helpers import make_client

from anvesh import demo, handoff

client = make_client()


def _wait(cid, timeout=30):
    t0 = time.time()
    while cid in demo.RUNNING and time.time() - t0 < timeout:
        time.sleep(0.1)
    assert cid not in demo.RUNNING, "demo did not finish"
    return client.get(f"/api/cases/{cid}").json()


def test_demo_scenarios_are_listed():
    keys = {s["key"] for s in client.get("/api/demo/scenarios").json()}
    assert {"digital_arrest", "tamil_crisis"} <= keys


def test_digital_arrest_demo_fills_the_case_and_starts_tele_manas(monkeypatch):
    monkeypatch.setattr(handoff, "COUNTDOWN_S", 600)  # keep the transfer pending while we look
    r = client.post("/api/demo/calls", json={"scenario": "digital_arrest", "speed": 20})
    assert r.status_code == 201, r.text
    c = _wait(r.json()["id"])
    assert c["synthetic"] is True and c["processing"] is None
    assert len(c["segments"]) == len(demo.SCENARIOS["digital_arrest"]["steps"])
    names = {f["name"] for f in c["fields"]}
    assert {"amount_lost", "upi_id", "bank"} <= names
    assert c["safety"]["crisis"] and c["safety"]["coercion"]
    tm = [h for h in c["handoffs"] if h["helpline"] == "telemanas"]
    assert tm and tm[0]["trigger"] == "auto"  # the demo counts as a live call
    assert c["id"] not in handoff.LIVE_CASES  # and stops counting as live when it ends


def test_tamil_demo_reaches_tele_manas_from_tamil_words(monkeypatch):
    monkeypatch.setattr(handoff, "COUNTDOWN_S", 600)
    r = client.post("/api/demo/calls", json={"scenario": "tamil_crisis", "speed": 20})
    c = _wait(r.json()["id"])
    assert c["language"] == "ta" and c["safety"]["crisis"]
    assert any(h["helpline"] == "telemanas" for h in c["handoffs"])


def test_unknown_demo_is_404():
    assert client.post("/api/demo/calls", json={"scenario": "nope_nope"}).status_code == 404
