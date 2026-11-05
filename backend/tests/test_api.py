
from anvesh.api import app  # noqa: F401
from helpers import ORIGIN, make_client  # noqa: F401
from anvesh.models import AuditEvent, SessionLocal

client = make_client()


def _new(mode="victim", **kw):
    r = client.post("/api/cases", json={"mode": mode, "caller_label": "Test caller", **kw})
    assert r.status_code == 201, r.text
    return r.json()


def _say(cid, *lines, speaker="caller"):
    segs = [{"start": i * 5.0, "end": i * 5.0 + 5, "text": t, "speaker": speaker} for i, t in enumerate(lines)]
    r = client.post(f"/api/cases/{cid}/segments", json={"segments": segs})
    assert r.status_code == 200, r.text
    return r.json()


def test_health_reports_models_honestly():
    r = client.get("/api/health").json()
    assert r["ok"] and {m["name"] for m in r["models"]} == {"asr", "asr_final", "asr_indic", "voice_emotion", "speaker_embedding",
                                                            "voice_authenticity"}
    assert all(not m["loaded"] for m in r["models"])  # disabled in tests


def test_call_to_case_file_with_evidence_and_triage():
    c = _new(district="Pune")
    d = _say(c["id"], "mere saadhe paanch hazaar kat gaye abhi abhi",
             "ramesh at the rate okaxis pe bheje, SBI se", "unhone video call pe rakha, kisi ko mat batana bola")
    fields = {f["name"]: f for f in d["fields"]}
    assert fields["amount_lost"]["value"] == "5500"
    assert fields["upi_id"]["evidence"][0]["quote"] == "ramesh at the rate okaxis"
    assert d["priority"] == "IMMEDIATE"
    assert d["safety"]["coercion"] is True
    assert d["copilot"], "co-pilot should suggest next questions"


def test_verify_and_correct_fields_are_audited_and_sticky():
    c = _new()
    _say(c["id"], "10000 rupaye kat gaye ramesh at the rate ybl ko")
    d = client.patch(f"/api/cases/{c['id']}/fields/upi_id", json={"action": "correct", "value": "ramesh1@ybl"}).json()
    f = {x["name"]: x for x in d["fields"]}
    assert f["upi_id"]["status"] == "corrected" and f["upi_id"]["value"] == "ramesh1@ybl"
    d = _say(c["id"], "aur kuch nahi sir")  # re-analysis must not overwrite the correction
    f = {x["name"]: x for x in d["fields"]}
    assert f["upi_id"]["value"] == "ramesh1@ybl"
    log = client.get(f"/api/audit?case_id={c['id']}").json()
    assert any(e["action"] == "field.correct" for e in log)


def test_ring_linker_links_cases_by_identifier():
    a = _new(district="Indore")
    _say(a["id"], "20000 rupaye bheje ringtest.mule at the rate ybl pe")
    b = _new(district="Patna")
    d = _say(b["id"], "maine 15000 bhej diye ringtest.mule at the rate ybl ko")
    kinds = {l["kind"] for l in d["links"]}
    assert "identifier" in kinds
    g = client.get("/api/graph").json()
    assert any(set(r["cases"]) >= {a["id"], b["id"]} for r in g["rings"])


def test_suspect_mode_has_tactics_and_no_distress():
    c = _new(mode="suspect_recording")
    d = _say(c["id"], "Main CBI se bol raha hoon, abhi turant paise bhejo, kisi ko mat batana", speaker="suspect")
    assert d["distress"]["enabled"] is False
    assert {t["key"] for t in d["tactics"]["items"]} >= {"authority_impersonation", "urgency", "secrecy_demand"}
    assert d["safety"]["coercion"] is False  # coercion detection is for victims, not suspects


def test_readback_and_report():
    c = _new()
    _say(c["id"], "48000 rupaye kat gaye ramesh at the rate okaxis ko, HDFC se, 2:10 baje")
    rb = client.get(f"/api/cases/{c['id']}/readback?lang=hi").json()
    assert "48,000" in rb["text"] and "sahi" in rb["text"]
    rep = client.get(f"/api/cases/{c['id']}/report").json()
    assert rep["integrity"]["audit_chain_ok"] is True and "does not determine guilt" in rep["disclaimer"]


def test_audit_chain_detects_tampering():
    assert client.get("/api/audit/verify").json()["ok"] is True
    with SessionLocal() as db:
        ev = db.query(AuditEvent).order_by(AuditEvent.id).first()
        original = ev.payload
        ev.payload = '{"tampered":true}'
        db.commit()
    v = client.get("/api/audit/verify").json()
    assert v["ok"] is False and v["broken_at"] == ev.id
    with SessionLocal() as db:  # restore so later tests start from an intact chain
        db.get(AuditEvent, ev.id).payload = original
        db.commit()
    assert client.get("/api/audit/verify").json()["ok"] is True


def test_upload_without_asr_fails_clearly():
    c = _new()
    r = client.post(f"/api/cases/{c['id']}/recordings", files={"file": ("a.wav", b"RIFF0000WAVE", "audio/wav")})
    assert r.status_code == 503 and "Speech-to-text" in r.json()["detail"]


def test_radar_groups_by_scam_type():
    c = _new(district="Thane")
    _say(c["id"], "bijli wale ne bola light connection kat jayega, 5000 rupaye bheje")
    r = client.get("/api/radar").json()
    assert any(g["key"] == "utility" and g["last_24h"] >= 1 and "Thane" in g["districts"] for g in r["groups"])
