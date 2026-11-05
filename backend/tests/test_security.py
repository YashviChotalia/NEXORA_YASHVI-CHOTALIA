"""Security tests: authentication, authorisation, CSRF, confidentiality at rest, audit integrity,
brute force, input limits, headers and WebSocket protections.

Each test names the threat it covers. They run in CI with every other test.
"""

import re
import sqlite3
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from helpers import ORIGIN, PASSWORD, make_client, make_user
from starlette.websockets import WebSocketDisconnect

from anvesh import audit, auth, keys, storage
from anvesh.api import app
from anvesh.config import settings
from anvesh.models import AuditEvent, AuthSession, Recording, SessionLocal, User, utcnow

PII = {"name": "Kavita Deshmukh", "phone": "9123409876", "upi": "kavita.d@okicici"}


def _case_with_pii(c: TestClient, mode="victim") -> str:
    case = c.post("/api/cases", json={"mode": mode, "caller_label": PII["name"], "district": "Pune"}).json()
    r = c.post(f"/api/cases/{case['id']}/segments", json={"segments": [{
        "start": 0, "end": 5, "speaker": "caller",
        "text": f"mera number {PII['phone']} hai, maine 40000 rupaye {PII['upi']} pe bheje aadha ghanta pehle"}]})
    assert r.status_code == 200, r.text
    return case["id"]


def _routes():
    for r in app.routes:
        for m in getattr(r, "methods", None) or []:
            if m not in ("HEAD", "OPTIONS"):
                yield m, r.path


PUBLIC = {("GET", "/api/health"), ("GET", "/api/auth/state"), ("POST", "/api/auth/login"),
          ("POST", "/api/auth/setup"), ("GET", "/openapi.json"), ("GET", "/docs"), ("GET", "/redoc"),
          ("GET", "/docs/oauth2-redirect")}


def _fill(path: str) -> str:
    return (path.replace("{case_id}", "AV-2026-00001").replace("{rec_id}", "REC-1").replace("{seg_id}", "SEG-1")
            .replace("{name}", "upi_id").replace("{key}", "k").replace("{link_id}", "1").replace("{user_id}", "1"))


# --------------------------------------------------------------------------- authentication

def test_every_route_requires_a_session_except_the_public_ones():
    """Threat: unauthenticated access to case data (previously every endpoint was open)."""
    anon = TestClient(app)
    checked = 0
    for method, path in _routes():
        if (method, path) in PUBLIC:
            continue
        r = anon.request(method, _fill(path), json={})
        assert r.status_code == 401, f"{method} {path} returned {r.status_code} without a session"
        checked += 1
    assert checked >= 25


def test_health_hides_details_from_anonymous_callers():
    assert set(TestClient(app).get("/api/health").json()) == {"ok"}


def test_actor_comes_from_the_session_not_a_header():
    """Threat: impersonation by sending X-Operator (previously trusted)."""
    c = make_client(("operator",))
    cid = c.post("/api/cases", json={"mode": "victim"}, headers={"X-Operator": "SP Sharma"}).json()["id"]
    with SessionLocal() as db:
        ev = db.query(AuditEvent).filter(AuditEvent.case_id == cid, AuditEvent.action == "case.created").one()
    assert c.username in ev.actor and "Sharma" not in ev.actor


def test_session_cookie_flags_and_token_stored_only_as_hash():
    u = make_user(("operator",))
    c = TestClient(app)
    r = c.post("/api/auth/login", json={"username": u, "password": PASSWORD})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    token = c.cookies.get(auth.COOKIE)
    assert token and len(token) >= 40
    with SessionLocal() as db:
        assert db.get(AuthSession, token) is None  # raw token is never a key in the database
        assert db.get(AuthSession, keys.token_hash(token)) is not None


def test_logout_revokes_the_session_server_side():
    c = make_client(("operator",))
    stolen = c.cookies.get(auth.COOKIE)
    assert c.post("/api/auth/logout").status_code == 200
    replay = TestClient(app)
    replay.cookies.set(auth.COOKIE, stolen)
    assert replay.get("/api/cases").status_code == 401


def test_idle_session_expires():
    c = make_client(("operator",))
    with SessionLocal() as db:
        s = db.get(AuthSession, keys.token_hash(c.cookies.get(auth.COOKIE)))
        s.last_seen = utcnow() - timedelta(minutes=auth.IDLE_MINUTES + 1)
        db.commit()
    assert c.get("/api/cases").status_code == 401


def test_passwords_are_hashed_with_scrypt_and_policy_is_enforced():
    u = make_user(("operator",))
    with SessionLocal() as db:
        h = db.query(User).filter(User.username == u).one().pw_hash
    assert h.startswith("scrypt$") and PASSWORD not in h
    admin = make_client(("admin",))
    weak = admin.post("/api/users", json={"username": "weakpw", "display_name": "W", "roles": ["operator"],
                                          "password": "password"})
    assert weak.status_code == 422


def test_brute_force_locks_the_account_and_gives_no_username_oracle():
    u = make_user(("operator",))
    c = TestClient(app)
    msgs = set()
    for _ in range(auth.LOCK_AFTER):
        r = c.post("/api/auth/login", json={"username": u, "password": "wrong-password-1A!"})
        msgs.add((r.status_code, r.json()["detail"]))
    unknown = c.post("/api/auth/login", json={"username": "nobody-here", "password": "x"})
    msgs.add((unknown.status_code, unknown.json()["detail"]))
    assert len(msgs) == 1  # same answer for a wrong password and an unknown user
    r = c.post("/api/auth/login", json={"username": u, "password": PASSWORD})
    assert r.status_code == 423  # locked even with the right password
    auth.limiter.reset()


def test_login_rate_limit_per_client():
    c = TestClient(app)
    codes = [c.post("/api/auth/login", json={"username": f"x{i}", "password": "y"}).status_code for i in range(12)]
    assert 429 in codes
    auth.limiter.reset()


def test_first_run_setup_cannot_be_repeated():
    make_user(("admin",))
    r = TestClient(app).post("/api/auth/setup", json={"username": "attacker", "display_name": "A",
                                                       "password": "Str0ng-Passw0rd!"})
    assert r.status_code == 409


def test_new_accounts_must_change_their_password_first():
    admin = make_client(("admin",))
    r = admin.post("/api/users", json={"username": "newop", "display_name": "New Op", "roles": ["operator"],
                                       "password": "Temp-Passw0rd-99"})
    assert r.status_code == 201
    c = TestClient(app)
    login = c.post("/api/auth/login", json={"username": "newop", "password": "Temp-Passw0rd-99"}).json()
    c.headers["X-CSRF-Token"] = login["csrf_token"]
    assert c.get("/api/cases").status_code == 403
    assert c.post("/api/auth/password", json={"current": "Temp-Passw0rd-99", "new": "Fresh-Passw0rd-77"}).status_code == 200
    assert c.get("/api/cases").status_code == 200


def test_disabling_a_user_kills_their_sessions():
    admin = make_client(("admin",))
    op = make_client(("operator",))
    uid = next(u["id"] for u in admin.get("/api/users").json() if u["username"] == op.username)
    assert admin.patch(f"/api/users/{uid}", json={"disabled": True}).status_code == 200
    assert op.get("/api/cases").status_code == 401


# --------------------------------------------------------------------------- authorisation

def test_role_matrix():
    """Threat: privilege escalation. Operators work cases; only supervisors review links; only admins manage users;
    admins cannot read case content (separation of duties)."""
    op, sup, adm = make_client(("operator",)), make_client(("supervisor",)), make_client(("admin",))
    cid = _case_with_pii(op)
    assert op.get(f"/api/cases/{cid}").status_code == 200
    assert adm.get(f"/api/cases/{cid}").status_code == 403
    assert adm.get("/api/cases").status_code == 403
    assert op.get("/api/users").status_code == 403
    assert sup.get("/api/users").status_code == 403
    assert op.get("/api/audit").status_code == 403
    assert sup.get("/api/audit").status_code == 200 and adm.get("/api/audit").status_code == 200
    assert op.patch("/api/links/1", json={"status": "confirmed"}).status_code == 403
    assert op.post(f"/api/cases/{cid}/erase", json={"reason": "testing erase"}).status_code == 403
    assert adm.post("/api/users", json={"username": "promo", "display_name": "P", "roles": ["root"],
                                        "password": "Str0ng-Passw0rd!"}).status_code == 422


def test_operator_cannot_create_hidden_training_cases():
    op = make_client(("operator",))
    c = op.post("/api/cases", json={"mode": "victim", "synthetic": True}).json()
    assert c["synthetic"] is False


def test_informant_identity_is_protected_from_other_operators():
    """Threat: exposure of an informant to staff without need-to-know."""
    taker, other, sup = make_client(("operator",)), make_client(("operator",)), make_client(("supervisor",))
    cid = _case_with_pii(taker, mode="informant")
    assert taker.get(f"/api/cases/{cid}").json()["caller_label"] == PII["name"]
    seen = other.get(f"/api/cases/{cid}").json()
    assert seen["caller_label"] == "Protected informant" and seen["identity_protected"]
    listed = next(c for c in other.get("/api/cases").json() if c["id"] == cid)
    assert listed["caller_label"] == "Protected informant"
    assert sup.get(f"/api/cases/{cid}").json()["caller_label"] == PII["name"]


# --------------------------------------------------------------------------- CSRF / CORS

def test_state_changes_need_the_csrf_token():
    c = make_client(("operator",))
    tok = c.headers.pop("X-CSRF-Token")
    assert c.post("/api/cases", json={"mode": "victim"}).status_code == 403
    assert c.post("/api/cases", json={"mode": "victim"}, headers={"X-CSRF-Token": "forged"}).status_code == 403
    assert c.post("/api/cases", json={"mode": "victim"}, headers={"X-CSRF-Token": tok}).status_code == 201


def test_cors_rejects_unknown_origins():
    r = TestClient(app).options("/api/cases", headers={"Origin": "https://evil.example",
                                                       "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
    ok = TestClient(app).options("/api/cases", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == ORIGIN
    assert "*" not in settings.cors_origins


# --------------------------------------------------------------------------- confidentiality at rest

def _raw_db_text() -> str:
    db_path = settings.database_url.removeprefix("sqlite:///")
    con = sqlite3.connect(db_path)
    out = []
    for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        for row in con.execute(f"SELECT * FROM {table}"):
            out.extend(str(v) for v in row)
    con.close()
    return "\n".join(out)


def test_personal_data_is_encrypted_in_the_database_file():
    """Threat: a copied database / backup exposes complainants. Names, numbers, UPI IDs, transcript text,
    extracted fields, identifiers and link reasons must not appear in plaintext anywhere in the DB."""
    c = make_client(("operator",))
    cid = _case_with_pii(c)
    detail = c.get(f"/api/cases/{cid}").json()
    assert detail["caller_label"] == PII["name"]  # readable through the API
    assert any(f["value"] == PII["upi"] for f in detail["fields"])
    raw = _raw_db_text()
    for label, value in PII.items():
        assert value not in raw, f"{label} found in plaintext in the database"
    assert "enc1:" in raw


def test_ciphertext_is_bound_to_its_column():
    """Threat: copying ciphertext from one column to another (e.g. into a field shown to other users)."""
    ct = keys.encrypt_str("secret", "segments.text")
    with pytest.raises(keys.DecryptionError):
        keys.decrypt_str(ct, "cases.caller_label")
    tampered = ct[:-4] + ("AAAA" if not ct.endswith("AAAA") else "BBBB")
    with pytest.raises(keys.DecryptionError):
        keys.decrypt_str(tampered, "segments.text")


def test_identifier_linking_works_on_blind_indexes_only():
    c = make_client(("operator",))
    a, b = _case_with_pii(c), _case_with_pii(c)
    assert any(l["other"]["id"] == b and l["kind"] == "identifier" for l in c.get(f"/api/cases/{a}").json()["links"])
    assert PII["upi"] not in _raw_db_text()


def test_master_key_lives_outside_the_data_directory():
    kd = keys.key_dir()
    assert settings.data_dir not in kd.parents and kd != settings.data_dir
    assert not (settings.data_dir / ".dev_data_key").exists()


def test_evidence_files_are_encrypted_authenticated_and_confined():
    sha = audit.sha256_bytes(b"RIFF....WAVEfake")
    path = storage.save(sha, b"RIFF....WAVEfake")
    blob = open(path, "rb").read()
    assert b"RIFF" not in blob and sha not in path  # encrypted, and the filename does not reveal the hash
    assert storage.load(path) == b"RIFF....WAVEfake"
    open(path, "wb").write(blob[:-3] + b"xyz")
    with pytest.raises(storage.EvidenceError):
        storage.load(path)
    with pytest.raises(storage.EvidenceError):
        storage.load(str(settings.data_dir / ".." / "outside.bin"))


def test_tampered_evidence_is_refused_and_logged():
    c = make_client(("operator",))
    cid = c.post("/api/cases", json={"mode": "victim"}).json()["id"]
    from anvesh import pipeline
    from anvesh.speech import audio as A
    import numpy as np
    wav = A.to_wav_bytes(np.zeros(16000, dtype=np.float32))
    with SessionLocal() as db:
        rec = pipeline.ingest_audio(db, pipeline.get_case(db, cid), wav, "x.wav", "audio/wav", "call", "t", 0,
                                    precomputed={})
        db.commit()
        rid, path = rec.id, rec.stored_path
    assert c.get(f"/api/recordings/{rid}/audio").status_code == 200
    data = bytearray(open(path, "rb").read())
    data[40] ^= 1
    open(path, "wb").write(bytes(data))
    assert c.get(f"/api/recordings/{rid}/audio").status_code == 409
    with SessionLocal() as db:
        assert db.query(AuditEvent).filter(AuditEvent.action == "evidence.integrity_failed").count() >= 1
        assert db.get(Recording, rid).filename == "x.wav"


# --------------------------------------------------------------------------- audit integrity & privacy

def test_audit_log_carries_no_personal_data():
    c = make_client(("supervisor",))
    cid = _case_with_pii(c)
    c.patch(f"/api/cases/{cid}/fields/upi_id", json={"action": "correct", "value": "other.person@ybl"})
    c.patch(f"/api/cases/{cid}", json={"caller_label": "Another Name"})
    blob = "\n".join(e["payload"] for e in c.get(f"/api/audit?case_id={cid}").json())
    for v in [*PII.values(), "other.person@ybl", "Another Name"]:
        assert v not in blob
    assert "hmac:" in blob


def test_audit_chain_detects_edits_and_cannot_be_reforged_without_the_key():
    c = make_client(("supervisor",))
    _case_with_pii(c)
    assert c.get("/api/audit/verify").json()["ok"]
    with SessionLocal() as db:
        ev = db.query(AuditEvent).filter(AuditEvent.alg == "hmac").order_by(AuditEvent.id.desc()).first()
        original = (ev.actor, ev.hash)
        ev.actor = "someone else"
        db.commit()
    assert not c.get("/api/audit/verify").json()["ok"]
    with SessionLocal() as db:  # attacker recomputes with plain SHA-256: still detected
        ev = db.get(AuditEvent, ev.id)
        import hashlib
        body = audit._body(ev.ts.isoformat(), ev.actor, ev.action, ev.case_id, ev.payload, ev.model_version)
        ev.hash = hashlib.sha256((ev.prev_hash + body).encode()).hexdigest()
        db.commit()
    assert not c.get("/api/audit/verify").json()["ok"]
    with SessionLocal() as db:
        ev = db.get(AuditEvent, ev.id)
        ev.actor, ev.hash = original
        db.commit()
    assert c.get("/api/audit/verify").json()["ok"]


def test_sensitive_reads_are_access_logged():
    c = make_client(("operator",))
    cid = _case_with_pii(c)
    c.get(f"/api/cases/{cid}")
    c.get(f"/api/cases/{cid}/report")
    with SessionLocal() as db:
        acts = {e.action for e in db.query(AuditEvent).filter(AuditEvent.case_id == cid)}
    assert {"case.viewed", "report.generated"} <= acts


def test_erasure_removes_data_and_evidence_but_keeps_the_audit_trail():
    sup = make_client(("supervisor",))
    cid = _case_with_pii(sup)
    assert sup.post(f"/api/cases/{cid}/erase", json={"reason": "wrong intake, duplicate"}).status_code == 200
    assert sup.get(f"/api/cases/{cid}").status_code == 404
    with SessionLocal() as db:
        assert db.query(AuditEvent).filter(AuditEvent.case_id == cid, AuditEvent.action == "case.erased").count() == 1
    assert sup.get("/api/audit/verify").json()["ok"]


# --------------------------------------------------------------------------- input handling & headers

def test_security_headers_and_no_caching_of_personal_data():
    c = make_client(("operator",))
    h = c.get("/api/cases").headers
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert h["cache-control"] == "no-store"
    assert "default-src 'none'" in h["content-security-policy"]
    assert h["referrer-policy"] == "no-referrer"


@pytest.mark.parametrize("bad", ["../../etc/passwd", "AV-2026-00001' OR '1'='1", "AV-2026-00001;DROP TABLE cases",
                                 "%2e%2e%2fsecret"])
def test_malformed_ids_are_rejected_before_reaching_the_database(bad):
    c = make_client(("operator",))
    assert c.get(f"/api/cases/{bad}").status_code in (404, 422)


def test_injection_strings_are_stored_as_data():
    c = make_client(("operator",))
    evil = "'); DROP TABLE cases; -- <script>alert(1)</script>"
    cid = c.post("/api/cases", json={"mode": "victim", "caller_label": evil}).json()["id"]
    assert c.get(f"/api/cases/{cid}").json()["caller_label"] == evil
    assert c.get("/api/cases").status_code == 200


def test_oversized_and_wrong_type_requests_are_refused():
    c = make_client(("operator",))
    cid = c.post("/api/cases", json={"mode": "victim"}).json()["id"]
    big = {"segments": [{"start": 0, "end": 1, "text": "x" * 4001}]}
    assert c.post(f"/api/cases/{cid}/segments", json=big).status_code == 422
    huge = "x" * (600 * 1024)
    assert c.post(f"/api/cases/{cid}/segments", content=huge,
                  headers={"content-type": "application/json"}).status_code == 413
    r = c.post(f"/api/cases/{cid}/recordings", files={"file": ("a.exe", b"MZ\x90\x00", "application/x-msdownload")})
    assert r.status_code == 415


def test_uploaded_filenames_are_sanitised():
    from anvesh.api import _safe_filename
    assert _safe_filename("..\\..\\windows\\system32\\evil.wav") == "evil.wav"
    assert _safe_filename("../../etc/passwd") == "passwd"
    assert "<" not in _safe_filename("<img src=x>.wav")


# --------------------------------------------------------------------------- WebSocket

def test_websocket_requires_session_and_allowed_origin():
    """Threat: cross-site WebSocket hijacking and anonymous audio injection."""
    c = make_client(("operator",))
    cid = c.post("/api/cases", json={"mode": "victim"}).json()["id"]
    with pytest.raises(WebSocketDisconnect) as e1:
        with TestClient(app).websocket_connect(f"/ws/cases/{cid}/live", headers={"origin": ORIGIN}) as ws:
            ws.receive_json()
    assert e1.value.code == 4401
    with pytest.raises(WebSocketDisconnect) as e2:
        with c.websocket_connect(f"/ws/cases/{cid}/live", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()
    assert e2.value.code == 4403
    admin = make_client(("admin",))
    with pytest.raises(WebSocketDisconnect) as e3:
        with admin.websocket_connect(f"/ws/cases/{cid}/live", headers={"origin": ORIGIN}) as ws:
            ws.receive_json()
    assert e3.value.code == 4401


def test_no_secrets_or_tokens_in_frontend_bundle_config():
    """Threat: a shared API token shipped to every browser (previously NEXT_PUBLIC_API_TOKEN)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "frontend"
    if not root.exists():
        pytest.skip("frontend not present")
    src = "\n".join(p.read_text(errors="ignore") for p in (root / "lib").glob("*.ts"))
    assert "NEXT_PUBLIC_API_TOKEN" not in src
    assert not re.search(r"localStorage\.(setItem|getItem)\([^)]*(token|csrf|password)", src, re.I)


def test_existing_plaintext_databases_are_encrypted_on_upgrade():
    """Upgrade path: rows written by the old version (plaintext) are encrypted in place and still readable."""
    from sqlalchemy import text

    from anvesh.models import Case, Entity, engine, migrate_plaintext
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO cases (id, created_at, updated_at, call_started_at, mode, channel, status, "
                          "caller_label, synthetic, priority, priority_score, playbook_conf, narrative, triage, "
                          "distress, safety, tactics, copilot, asked, windows) VALUES ('AV-2025-L0001', '2025-01-01', "
                          "'2025-01-01', '2025-01-01', 'victim', 'helpline', 'open', 'Legacy Person', 0, 'STANDARD', 0, "
                          "0, 'old narrative', '{\"headline\": \"h\"}', '{}', '{}', '{}', '[]', '[]', '[]')"))
        conn.execute(text("INSERT INTO entities (case_id, type, value) VALUES ('AV-2025-L0001', 'upi_id', 'legacy@ybl')"))
    assert "Legacy Person" in _raw_db_text()
    migrate_plaintext()
    raw = _raw_db_text()
    assert "Legacy Person" not in raw and "legacy@ybl" not in raw and "old narrative" not in raw
    with SessionLocal() as db:
        c = db.get(Case, "AV-2025-L0001")
        assert c.caller_label == "Legacy Person" and c.triage == {"headline": "h"}
        e = db.query(Entity).filter(Entity.case_id == c.id).one()
        assert e.value == "legacy@ybl" and e.value_hash == keys.blind_index("upi_id", "legacy@ybl")


def test_csrf_token_survives_a_second_tab():
    """Opening the app in another tab (auth/state) must not break the first tab's CSRF token."""
    c = make_client(("operator",))
    first = c.headers["X-CSRF-Token"]
    other_tab = c.get("/api/auth/state").json()["csrf_token"]
    assert other_tab == first
    assert c.post("/api/cases", json={"mode": "victim"}).status_code == 201


def test_bootstrap_admin_only_when_no_accounts(monkeypatch):
    from anvesh.api import bootstrap_admin
    from anvesh.models import SessionLocal, User
    monkeypatch.setenv("ANVESH_BOOTSTRAP_ADMIN", "boot.admin")
    monkeypatch.setenv("ANVESH_BOOTSTRAP_PASSWORD", "weak")
    with SessionLocal() as db:
        empty = auth.user_count(db) == 0
    if empty:  # a weak password is refused even on an empty database
        assert bootstrap_admin() is None
    make_user(("admin",))
    monkeypatch.setenv("ANVESH_BOOTSTRAP_PASSWORD", "Str0ng-Boot-Passw0rd!")
    assert bootstrap_admin() is None  # accounts exist: never adds a second admin
    with SessionLocal() as db:
        assert db.query(User).filter_by(username="boot.admin").first() is None


def test_bootstrap_admin_must_change_password(monkeypatch):
    from anvesh.api import bootstrap_admin
    from anvesh.models import SessionLocal, User
    monkeypatch.setattr(auth, "user_count", lambda db: 0)
    monkeypatch.setenv("ANVESH_BOOTSTRAP_ADMIN", "boot.admin2")
    monkeypatch.setenv("ANVESH_BOOTSTRAP_PASSWORD", "Str0ng-Boot-Passw0rd!")
    assert bootstrap_admin() == "boot.admin2"
    with SessionLocal() as db:
        u = db.query(User).filter_by(username="boot.admin2").one()
        assert u.must_change_password and set(u.roles) == {"admin", "supervisor"}
