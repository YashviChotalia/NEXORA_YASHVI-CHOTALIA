"""Database schema (SQLAlchemy 2.0). SQLite for development, PostgreSQL in deployment."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine,
                        event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.types import TypeDecorator

from . import keys
from .config import settings


class EncryptedText(TypeDecorator):
    """Text column stored as AES-256-GCM ciphertext; the column name is bound as associated data."""
    impl = Text
    cache_ok = True

    def __init__(self, context: str):
        super().__init__()
        self.context = context

    def process_bind_param(self, value, dialect):
        return keys.encrypt_str(value, self.context) if value is not None else None

    def process_result_value(self, value, dialect):
        return keys.decrypt_str(value, self.context)


class EncryptedJSON(TypeDecorator):
    """JSON value stored encrypted (same scheme as EncryptedText)."""
    impl = Text
    cache_ok = True

    def __init__(self, context: str):
        super().__init__()
        self.context = context

    def process_bind_param(self, value, dialect):
        return None if value is None else keys.encrypt_str(json.dumps(value, ensure_ascii=False), self.context)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        plain = keys.decrypt_str(value, self.context)
        return json.loads(plain) if isinstance(plain, str) else plain


# (table, column) pairs that hold personal or case-sensitive data; used by the plaintext migration
ENCRYPTED_COLUMNS: list[tuple[str, str]] = []


def _enc(table: str, column: str, json_: bool = False):
    ENCRYPTED_COLUMNS.append((table, column))
    ctx = f"{table}.{column}"
    return EncryptedJSON(ctx) if json_ else EncryptedText(ctx)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    type_annotation_map = {dict: JSON, list: JSON}


class Case(Base):
    __tablename__ = "cases"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    call_started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    mode: Mapped[str] = mapped_column(String(32))  # victim | suspect_recording | informant
    channel: Mapped[str] = mapped_column(String(32), default="helpline")
    status: Mapped[str] = mapped_column(String(16), default="open")
    caller_label: Mapped[str] = mapped_column(_enc("cases", "caller_label"), default="Unnamed caller")
    district: Mapped[str | None] = mapped_column(String(80), nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False)  # True = scripted training scenario, not a real call
    scenario: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # official basis for a training scenario
    created_by: Mapped[str | None] = mapped_column(String(64), nullable=True)  # username of the intake officer
    voice_auth: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # latest synthetic-voice assessment
    priority: Mapped[str] = mapped_column(String(16), default="STANDARD")
    priority_score: Mapped[int] = mapped_column(Integer, default=0)
    playbook_key: Mapped[str | None] = mapped_column(String(32), nullable=True)
    playbook_conf: Mapped[float] = mapped_column(Float, default=0.0)
    narrative: Mapped[str] = mapped_column(_enc("cases", "narrative"), default="")
    triage: Mapped[dict] = mapped_column(_enc("cases", "triage", True), default=dict)
    distress: Mapped[dict] = mapped_column(_enc("cases", "distress", True), default=dict)
    safety: Mapped[dict] = mapped_column(_enc("cases", "safety", True), default=dict)
    tactics: Mapped[dict] = mapped_column(_enc("cases", "tactics", True), default=dict)
    copilot: Mapped[list] = mapped_column(JSON, default=list)
    asked: Mapped[list] = mapped_column(JSON, default=list)
    windows: Mapped[list] = mapped_column(_enc("cases", "windows", True), default=list)
    intake_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    processing: Mapped[str | None] = mapped_column(String(120), nullable=True)  # background work in progress

    segments: Mapped[list["Segment"]] = relationship(back_populates="case", cascade="all, delete-orphan",
                                                     order_by="Segment.start")
    fields: Mapped[list["FieldRow"]] = relationship(back_populates="case", cascade="all, delete-orphan")
    recordings: Mapped[list["Recording"]] = relationship(back_populates="case", cascade="all, delete-orphan")


class Recording(Base):
    __tablename__ = "recordings"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"))
    kind: Mapped[str] = mapped_column(String(24), default="call")  # call | suspect_audio | live
    filename: Mapped[str] = mapped_column(_enc("recordings", "filename"))
    mime: Mapped[str] = mapped_column(String(64), default="audio/wav")
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    stored_path: Mapped[str] = mapped_column(String(512))
    speaker_embedding: Mapped[list | None] = mapped_column(_enc("recordings", "speaker_embedding", True), nullable=True)
    offset: Mapped[float] = mapped_column(Float, default=0.0)
    voice_auth: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # synthetic-voice assessment of this audio  # where this audio starts on the case timeline (s)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    case: Mapped[Case] = relationship(back_populates="recordings")


class Segment(Base):
    __tablename__ = "segments"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    recording_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(_enc("segments", "text"))
    text_native: Mapped[str | None] = mapped_column(_enc("segments", "text_native"), nullable=True)  # original script from ASR (e.g. Devanagari)
    speaker: Mapped[str | None] = mapped_column(String(24), nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    avg_logprob: Mapped[float | None] = mapped_column(Float, nullable=True)
    no_speech_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    low_confidence: Mapped[bool] = mapped_column(Boolean, default=False)
    words: Mapped[list | None] = mapped_column(_enc("segments", "words", True), nullable=True)
    case: Mapped[Case] = relationship(back_populates="segments")


class FieldRow(Base):
    __tablename__ = "fields"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    name: Mapped[str] = mapped_column(String(40))
    value: Mapped[str] = mapped_column(_enc("fields", "value"))
    display: Mapped[str] = mapped_column(_enc("fields", "display"))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16), default="predicted")
    evidence: Mapped[list] = mapped_column(_enc("fields", "evidence", True), default=list)
    note: Mapped[str | None] = mapped_column(_enc("fields", "note"), nullable=True)
    model_value: Mapped[str | None] = mapped_column(_enc("fields", "model_value"), nullable=True)
    updated_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    case: Mapped[Case] = relationship(back_populates="fields")


class Entity(Base):
    __tablename__ = "entities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    type: Mapped[str] = mapped_column(String(24), index=True)
    value: Mapped[str] = mapped_column(_enc("entities", "value"))
    value_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)  # keyed blind index


class Link(Base):
    __tablename__ = "links"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_a: Mapped[str] = mapped_column(String(32), index=True)
    case_b: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # identifier | script | voice
    score: Mapped[float] = mapped_column(Float)
    detail: Mapped[dict] = mapped_column(_enc("links", "detail", True), default=dict)
    status: Mapped[str] = mapped_column(String(16), default="candidate")  # candidate | confirmed | rejected
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    actor: Mapped[str] = mapped_column(String(80))
    action: Mapped[str] = mapped_column(String(64))
    case_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    payload: Mapped[str] = mapped_column(Text, default="{}")  # canonical JSON string
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(String(120), nullable=True)
    alg: Mapped[str | None] = mapped_column(String(16), nullable=True)  # None = legacy sha256, "hmac" = keyed


class Handoff(Base):
    """A warm transfer of a live call to a helpline (Tele-MANAS 14416, 112, 181, 1098, 1930). See handoff.py."""
    __tablename__ = "handoffs"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(32), ForeignKey("cases.id"), index=True)
    helpline: Mapped[str] = mapped_column(String(16))
    number: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(_enc("handoffs", "reason"), nullable=True)  # quotes the caller
    status: Mapped[str] = mapped_column(String(16))  # suggested | pending | connecting | connected | cancelled | dismissed | failed
    trigger: Mapped[str] = mapped_column(String(16))  # auto | suggest | officer
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    execute_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    actor: Mapped[str | None] = mapped_column(String(80), nullable=True)
    call_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[str | None] = mapped_column(String(240), nullable=True)
    evidence: Mapped[list | None] = mapped_column(JSON, nullable=True)  # segment ids and times of the words


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120))
    roles: Mapped[list] = mapped_column(JSON, default=list)
    pw_hash: Mapped[str] = mapped_column(String(255))
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False)
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)  # HMAC of the cookie token
    csrf_hash: Mapped[str] = mapped_column(String(160))  # CSRF token, encrypted (enc1:)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    client: Mapped[str | None] = mapped_column(String(64), nullable=True)


engine = create_engine(settings.database_url, future=True,
                       connect_args={"check_same_thread": False, "timeout": 30}
                       if settings.database_url.startswith("sqlite") else {})


if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _):  # pragma: no cover - driver hook
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)
    _add_missing_columns()
    migrate_plaintext()


def migrate_plaintext() -> int:
    """Encrypt any rows written before field encryption existed, and backfill blind indexes. Idempotent."""
    from sqlalchemy import text
    changed = 0
    with engine.begin() as conn:
        for table, col in ENCRYPTED_COLUMNS:
            # identifiers come from the schema constant above, never from input; validated anyway
            assert re.fullmatch(r"[a-z_]+", table) and re.fullmatch(r"[a-z_]+", col)
            rows = conn.execute(text(f'SELECT id, "{col}" FROM {table} WHERE "{col}" IS NOT NULL '  # nosec B608
                                     f'AND CAST("{col}" AS TEXT) NOT LIKE :p'), {"p": keys.PREFIX + "%"}).all()
            for rid, raw in rows:
                raw = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                conn.execute(text(f'UPDATE {table} SET "{col}" = :v WHERE id = :i'),  # nosec B608
                             {"v": keys.encrypt_str(raw, f"{table}.{col}"), "i": rid})
                changed += 1
        for rid, typ, val in conn.execute(text("SELECT id, type, value FROM entities WHERE value_hash IS NULL")).all():
            plain = keys.decrypt_str(val, "entities.value")
            conn.execute(text("UPDATE entities SET value_hash = :h WHERE id = :i"),
                         {"h": keys.blind_index(typ, plain or ""), "i": rid})
            changed += 1
    return changed


def _add_missing_columns() -> None:
    """Tiny forward-only migration: add columns introduced after a database was created."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in have:
                    ddl = col.type.compile(dialect=engine.dialect)
                    default = col.default.arg if col.default is not None and not callable(col.default.arg) else None
                    extra = f" DEFAULT {default!r}" if isinstance(default, (int, float)) else ""
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {ddl}{extra}'))


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
