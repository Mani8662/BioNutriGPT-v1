"""Small persistent account and risk-history store for single-instance deployments."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
import uuid
from typing import Any


SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
PASSWORD_ITERATIONS = 310_000
DB_PATH = Path(os.getenv("BIONUTRIGPT_DATA_DIR", Path(__file__).resolve().parent / "data")) / "bionutrigpt.sqlite3"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA busy_timeout = 10000")
    return db


@contextmanager
def _connection():
    db = _connect()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_store() -> None:
    with _connection() as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS accounts (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            );
            CREATE TABLE IF NOT EXISTS account_sessions (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                expires_at INTEGER NOT NULL,
                revoked_at INTEGER
            );
            CREATE INDEX IF NOT EXISTS account_sessions_account_idx ON account_sessions(account_id);
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                created_epoch INTEGER NOT NULL,
                source TEXT NOT NULL,
                diabetes REAL NOT NULL CHECK(diabetes >= 0 AND diabetes <= 1),
                obesity REAL NOT NULL CHECK(obesity >= 0 AND obesity <= 1),
                cvd REAL NOT NULL CHECK(cvd >= 0 AND cvd <= 1)
            );
            CREATE INDEX IF NOT EXISTS risk_assessments_timeline_idx
                ON risk_assessments(account_id, created_epoch);
            CREATE TABLE IF NOT EXISTS auth_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        account_columns = {str(row["name"]) for row in db.execute("PRAGMA table_info(accounts)").fetchall()}
        if "role" not in account_columns:
            db.execute("ALTER TABLE accounts ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        if not os.getenv("BIONUTRIGPT_AUTH_SECRET"):
            db.execute(
                "INSERT OR IGNORE INTO auth_meta(key, value) VALUES('signing_secret', ?)",
                (secrets.token_hex(32),),
            )


def _secret() -> bytes:
    configured = os.getenv("BIONUTRIGPT_AUTH_SECRET")
    if configured:
        return configured.encode("utf-8")
    with _connection() as db:
        row = db.execute("SELECT value FROM auth_meta WHERE key='signing_secret'").fetchone()
        if row is None:
            raise RuntimeError("Account signing secret is not initialized.")
        return str(row["value"]).encode("utf-8")


def _encode_token(account_id: str, session_id: str, expires_at: int) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(
        {"sub": account_id, "sid": session_id, "exp": expires_at}, separators=(",", ":")
    ).encode()).decode().rstrip("=")
    signature = hmac.new(_secret(), payload.encode(), hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    return f"{payload}.{encoded_signature}"


def _new_session(db: sqlite3.Connection, account_id: str) -> str:
    session_id = uuid.uuid4().hex
    expires_at = int(time.time()) + SESSION_TTL_SECONDS
    db.execute(
        "INSERT INTO account_sessions(id, account_id, expires_at) VALUES (?, ?, ?)",
        (session_id, account_id, expires_at),
    )
    return _encode_token(account_id, session_id, expires_at)


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)


def _public_account(row: sqlite3.Row) -> dict[str, str]:
    email = str(row["email"])
    admins = {item.strip().lower() for item in os.getenv("BIONUTRIGPT_ADMIN_EMAILS", "").split(",") if item.strip()}
    role = "admin" if email in admins or row["role"] == "admin" else "user"
    return {"id": str(row["id"]), "email": email, "created_at": str(row["created_at"]), "role": role}


def register(email: str, password: str) -> tuple[dict[str, str], str]:
    clean_email = email.strip().lower()
    if len(password) < 10 or len(password) > 128:
        raise ValueError("Password must be between 10 and 128 characters.")
    if len(clean_email) > 254 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", clean_email):
        raise ValueError("Enter a valid email address.")
    salt = secrets.token_bytes(16)
    now = datetime.now(timezone.utc).isoformat()
    account_id = uuid.uuid4().hex
    admins = {item.strip().lower() for item in os.getenv("BIONUTRIGPT_ADMIN_EMAILS", "").split(",") if item.strip()}
    role = "admin" if clean_email in admins else "user"
    try:
        with _connection() as db:
            db.execute(
                "INSERT INTO accounts(id, email, password_hash, password_salt, created_at, role) VALUES (?, ?, ?, ?, ?, ?)",
                (account_id, clean_email, _password_hash(password, salt).hex(), salt.hex(), now, role),
            )
            token = _new_session(db, account_id)
            row = db.execute("SELECT * FROM accounts WHERE id=?", (account_id,)).fetchone()
    except sqlite3.IntegrityError as exc:
        raise ValueError("An account with this email already exists.") from exc
    return _public_account(row), token


def login(email: str, password: str) -> tuple[dict[str, str], str]:
    with _connection() as db:
        row = db.execute("SELECT * FROM accounts WHERE email=?", (email.strip().lower(),)).fetchone()
        if row is None:
            # Do comparable work for unknown accounts to reduce email enumeration by timing.
            _password_hash(password, b"bionutrigpt-dummy-salt")
            raise ValueError("Email or password is incorrect.")
        expected = bytes.fromhex(str(row["password_hash"]))
        actual = _password_hash(password, bytes.fromhex(str(row["password_salt"])))
        if not hmac.compare_digest(expected, actual):
            raise ValueError("Email or password is incorrect.")
        token = _new_session(db, str(row["id"]))
    return _public_account(row), token


def authenticate(token: str | None) -> dict[str, str] | None:
    if not token or "." not in token:
        return None
    payload, signature = token.rsplit(".", 1)
    expected_sig = base64.urlsafe_b64encode(
        hmac.new(_secret(), payload.encode(), hashlib.sha256).digest()
    ).decode().rstrip("=")
    if not hmac.compare_digest(expected_sig, signature):
        return None
    try:
        decoded = payload + "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(decoded.encode()))
        account_id, session_id, expires = str(data["sub"]), str(data["sid"]), int(data["exp"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if expires <= int(time.time()):
        return None
    with _connection() as db:
        row = db.execute(
            """SELECT a.* FROM accounts a JOIN account_sessions s ON s.account_id=a.id
               WHERE a.id=? AND s.id=? AND s.revoked_at IS NULL AND s.expires_at>?""",
            (account_id, session_id, int(time.time())),
        ).fetchone()
    return _public_account(row) if row else None


def revoke(token: str | None) -> None:
    if not token or "." not in token:
        return
    try:
        payload = token.rsplit(".", 1)[0]
        payload += "=" * (-len(payload) % 4)
        session_id = str(json.loads(base64.urlsafe_b64decode(payload.encode()))["sid"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return
    with _connection() as db:
        db.execute("UPDATE account_sessions SET revoked_at=? WHERE id=?", (int(time.time()), session_id))


def add_risk_assessment(account_id: str, source: str, probabilities: dict[str, float]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    record = {
        "id": uuid.uuid4().hex,
        "created_at": now.isoformat(),
        "source": source,
        "probabilities": {key: float(probabilities[key]) for key in ("diabetes", "obesity", "cvd")},
    }
    with _connection() as db:
        db.execute(
            """INSERT INTO risk_assessments
               (id, account_id, created_at, created_epoch, source, diabetes, obesity, cvd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (record["id"], account_id, record["created_at"], int(now.timestamp()), source,
             record["probabilities"]["diabetes"], record["probabilities"]["obesity"], record["probabilities"]["cvd"]),
        )
    return record


def list_risk_assessments(account_id: str, limit: int = 250) -> list[dict[str, Any]]:
    with _connection() as db:
        rows = db.execute(
            """SELECT id, created_at, source, diabetes, obesity, cvd
               FROM risk_assessments WHERE account_id=? ORDER BY created_epoch DESC, rowid DESC LIMIT ?""",
            (account_id, limit),
        ).fetchall()
    items = [{
        "id": str(row["id"]), "created_at": str(row["created_at"]), "source": str(row["source"]),
        "probabilities": {key: float(row[key]) for key in ("diabetes", "obesity", "cvd")},
    } for row in rows]
    return list(reversed(items))


def delete_risk_assessment(account_id: str, assessment_id: str) -> bool:
    with _connection() as db:
        cursor = db.execute(
            "DELETE FROM risk_assessments WHERE id=? AND account_id=?", (assessment_id, account_id)
        )
    return cursor.rowcount > 0


def get_admin_summary() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    cutoff_epoch = int((now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=6)).timestamp())
    configured_admins = {item.strip().lower() for item in os.getenv("BIONUTRIGPT_ADMIN_EMAILS", "").split(",") if item.strip()}
    with _connection() as db:
        total_accounts = int(db.execute("SELECT COUNT(*) FROM accounts").fetchone()[0])
        account_roles = db.execute("SELECT email, role FROM accounts").fetchall()
        admin_accounts = sum(1 for row in account_roles if row["role"] == "admin" or row["email"] in configured_admins)
        total_assessments = int(db.execute("SELECT COUNT(*) FROM risk_assessments").fetchone()[0])
        recent_assessments = int(db.execute(
            "SELECT COUNT(*) FROM risk_assessments WHERE created_epoch>=?", (cutoff_epoch,)
        ).fetchone()[0])
        daily_rows = db.execute(
            """SELECT date(created_at) AS day, COUNT(*) AS count FROM risk_assessments
               WHERE created_epoch>=? GROUP BY date(created_at) ORDER BY day""",
            (cutoff_epoch,),
        ).fetchall()
        activity_rows = db.execute(
            "SELECT created_at, source FROM risk_assessments ORDER BY created_epoch DESC, rowid DESC LIMIT 12"
        ).fetchall()
    return {
        "total_accounts": total_accounts,
        "admin_accounts": admin_accounts,
        "total_assessments": total_assessments,
        "assessments_last_7_days": recent_assessments,
        "daily_assessments": [{"date": str(row["day"]), "count": int(row["count"])} for row in daily_rows],
        # Activity omits emails, account IDs, and risk values.
        "recent_activity": [{"created_at": str(row["created_at"]), "source": str(row["source"])} for row in activity_rows],
    }


init_store()
