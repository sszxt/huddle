"""Users, sign-in sessions and chat history, in one SQLite file.

SQLite because a node already runs with no services beside it, and a handful
of people chatting on a LAN is far inside what one file handles. One
connection, used from worker threads under a lock: calls are short, and a
lock keeps "count the users, then insert" (the first user becomes admin)
atomic without further ceremony.

Sessions are stored by the SHA-256 of their token, so a copy of the database
does not hand out working sign-ins. Chats are stored whole, as the web UI
keeps them: the server never needs to look inside one.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

log = logging.getLogger("huddle.accounts")

Role = Literal["admin", "user", "pending"]

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    role TEXT NOT NULL CHECK (role IN ('admin', 'user', 'pending')),
    password_hash TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    last_active_at INTEGER
);
CREATE TABLE sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);
CREATE INDEX sessions_by_user ON sessions(user_id);
CREATE TABLE chats (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    id TEXT NOT NULL,
    title TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, id)
);
CREATE INDEX chats_by_recency ON chats(user_id, updated_at DESC);
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

USER_COLUMNS = "id, name, email, role, created_at, last_active_at"


class User(BaseModel):
    id: str
    name: str
    email: str
    role: Role
    created_at: int
    last_active_at: int | None = None

    @property
    def active(self) -> bool:
        """Approved: may chat and keep history. A pending user may only wait."""
        return self.role in ("admin", "user")


class AuthSettings(BaseModel):
    """What an admin decides about sign-ups, as Open WebUI's admin panel does."""

    signup_enabled: bool = True
    # What a new sign-up becomes. Pending waits for an admin to approve it.
    default_role: Literal["pending", "user"] = "pending"


class AccountError(Exception):
    """A request the store refuses, with the HTTP status that describes it."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class SetupCodeError(AccountError):
    """The first account was asked for from another device without the right setup code."""

    def __init__(self) -> None:
        super().__init__(
            403,
            "the first account needs the setup code from the PC running Huddle: "
            "run `huddle admin-code` there, or create the account from that PC",
        )


# No 0/O/1/I/L: the code is read off a terminal and typed into a form.
SETUP_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
SETUP_CODE_LENGTH = 8


def _new_setup_code() -> str:
    raw = "".join(secrets.choice(SETUP_ALPHABET) for _ in range(SETUP_CODE_LENGTH))
    return f"{raw[:4]}-{raw[4:]}"


def _normalise_code(code: str) -> str:
    return "".join(ch for ch in code.upper() if ch.isalnum())


def _now() -> int:
    return int(time.time())


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class AccountStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------------

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.path.exists()
        conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        if fresh:
            # Password hashes and everyone's chats: readable by this user only.
            with contextlib.suppress(OSError):
                os.chmod(self.path, 0o600)
        self._conn = conn
        with self._tx() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self.path} was written by a newer Huddle (schema {version}); upgrade Huddle"
                )
            if version < 1:
                for statement in SCHEMA.split(";"):
                    if statement.strip():
                        db.execute(statement)
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            db.execute("DELETE FROM sessions WHERE expires_at <= ?", (_now(),))

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @contextlib.contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One transaction, holding the lock throughout."""
        with self._lock:
            if self._conn is None:
                raise AccountError(503, "the accounts database is not open")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    # -- users --------------------------------------------------------------------

    def count_users(self) -> int:
        with self._tx() as db:
            return int(db.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def sign_up(
        self,
        name: str,
        email: str,
        password_hash: str,
        *,
        setup_code: str | None = None,
        trusted: bool = True,
    ) -> User:
        """Create an account from the sign-up page.

        The first account is the admin, whatever the settings say: someone has
        to be able to approve the others. After that, sign-ups follow the
        admin's settings.

        Becoming admin is the one thing here worth stealing, and on a network
        that trusts its members the first visitor to the page would win it. So
        unless the request is ``trusted`` (it came from the PC itself), the first
        account must present the setup code that PC printed. The check shares a
        transaction with deciding who is first, so two sign-ups cannot both pass.
        """
        with self._tx() as db:
            first = db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
            settings = self._settings(db)
            if not first and not settings.signup_enabled:
                raise AccountError(403, "new sign-ups are turned off; ask the admin for an account")
            if first and not trusted:
                expected = self._setup_code(db)
                given = _normalise_code(setup_code or "")
                if not hmac.compare_digest(given.encode(), _normalise_code(expected).encode()):
                    raise SetupCodeError
            role: Role = "admin" if first else settings.default_role
            user = self._insert_user(db, name, email, password_hash, role)
            if first:
                db.execute("DELETE FROM settings WHERE key = 'setup_code'")
            return user

    def setup_code(self) -> str | None:
        """The code the first account needs from another device; None once one exists.

        Made when first asked for and kept until the first account is created,
        so it survives a restart between printing it and using it.
        """
        with self._tx() as db:
            if db.execute("SELECT COUNT(*) FROM users").fetchone()[0] != 0:
                return None
            return self._setup_code(db)

    def _setup_code(self, db: sqlite3.Connection) -> str:
        row = db.execute("SELECT value FROM settings WHERE key = 'setup_code'").fetchone()
        if row is not None:
            return str(row["value"])
        code = _new_setup_code()
        db.execute("INSERT INTO settings (key, value) VALUES ('setup_code', ?)", (code,))
        return code

    def add_user(self, name: str, email: str, password_hash: str, role: Role) -> User:
        """Create an account on an admin's behalf."""
        with self._tx() as db:
            return self._insert_user(db, name, email, password_hash, role)

    def _insert_user(
        self, db: sqlite3.Connection, name: str, email: str, password_hash: str, role: Role
    ) -> User:
        if db.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            raise AccountError(409, "an account with that email already exists")
        user = User(id=uuid.uuid4().hex, name=name, email=email, role=role, created_at=_now())
        db.execute(
            "INSERT INTO users (id, name, email, role, password_hash, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (user.id, user.name, user.email, user.role, password_hash, user.created_at),
        )
        return user

    def login_record(self, email: str) -> tuple[User, str] | None:
        """The account and password hash for an email, for checking a sign-in."""
        with self._tx() as db:
            row = db.execute(
                f"SELECT {USER_COLUMNS}, password_hash FROM users WHERE email = ?", (email,)
            ).fetchone()
        if row is None:
            return None
        return _user(row), str(row["password_hash"])

    def password_hash(self, user_id: str) -> str | None:
        with self._tx() as db:
            row = db.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
        return str(row["password_hash"]) if row else None

    def get_user(self, user_id: str) -> User | None:
        with self._tx() as db:
            row = _find(db, user_id)
        return _user(row) if row else None

    def list_users(self) -> list[User]:
        with self._tx() as db:
            rows = db.execute(f"SELECT {USER_COLUMNS} FROM users ORDER BY created_at").fetchall()
        return [_user(row) for row in rows]

    def update_user(
        self,
        user_id: str,
        *,
        name: str | None = None,
        email: str | None = None,
        role: Role | None = None,
        password_hash: str | None = None,
    ) -> User:
        with self._tx() as db:
            row = _find(db, user_id)
            if row is None:
                raise AccountError(404, "no such user")
            user = _user(row)
            if role is not None and role != "admin" and user.role == "admin":
                self._keep_an_admin(db, user_id)
            if email is not None and email.lower() != user.email.lower():
                taken = db.execute(
                    "SELECT 1 FROM users WHERE email = ? AND id != ?", (email, user_id)
                ).fetchone()
                if taken:
                    raise AccountError(409, "an account with that email already exists")
            changes = {"name": name, "email": email, "role": role, "password_hash": password_hash}
            for column, value in changes.items():
                if value is not None:
                    db.execute(f"UPDATE users SET {column} = ? WHERE id = ?", (value, user_id))
            if role == "pending" or password_hash is not None:
                # Held sign-ins no longer match what the admin decided.
                db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            updated = _find(db, user_id)
        assert updated is not None
        return _user(updated)

    def delete_user(self, user_id: str) -> None:
        with self._tx() as db:
            row = db.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise AccountError(404, "no such user")
            if row["role"] == "admin":
                self._keep_an_admin(db, user_id)
            # Sessions and chats go with the user (ON DELETE CASCADE).
            db.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def _keep_an_admin(self, db: sqlite3.Connection, leaving: str) -> None:
        others = db.execute(
            "SELECT COUNT(*) FROM users WHERE role = 'admin' AND id != ?", (leaving,)
        ).fetchone()[0]
        if others == 0:
            raise AccountError(409, "this is the only admin; make someone else admin first")

    # -- sessions -----------------------------------------------------------------

    def create_session(self, user_id: str, lifetime: float) -> str:
        token = secrets.token_urlsafe(32)
        now = _now()
        with self._tx() as db:
            db.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at)"
                " VALUES (?, ?, ?, ?)",
                (token_hash(token), user_id, now, now + int(lifetime)),
            )
            db.execute("UPDATE users SET last_active_at = ? WHERE id = ?", (now, user_id))
        return token

    def session_user(self, token: str) -> User | None:
        with self._tx() as db:
            row = db.execute(
                f"SELECT {', '.join('u.' + c for c in USER_COLUMNS.split(', '))}"
                " FROM sessions s JOIN users u ON u.id = s.user_id"
                " WHERE s.token_hash = ? AND s.expires_at > ?",
                (token_hash(token), _now()),
            ).fetchone()
        return _user(row) if row else None

    def touch(self, user_id: str) -> None:
        with self._tx() as db:
            db.execute("UPDATE users SET last_active_at = ? WHERE id = ?", (_now(), user_id))

    def delete_session(self, token: str) -> None:
        with self._tx() as db:
            db.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash(token),))

    # -- settings -----------------------------------------------------------------

    def settings(self) -> AuthSettings:
        with self._tx() as db:
            return self._settings(db)

    def save_settings(self, settings: AuthSettings) -> AuthSettings:
        with self._tx() as db:
            db.execute(
                "INSERT INTO settings (key, value) VALUES ('auth', ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (settings.model_dump_json(),),
            )
        return settings

    def _settings(self, db: sqlite3.Connection) -> AuthSettings:
        row = db.execute("SELECT value FROM settings WHERE key = 'auth'").fetchone()
        if row is None:
            return AuthSettings()
        try:
            return AuthSettings.model_validate_json(row["value"])
        except ValueError:
            log.warning("accounts: stored sign-up settings are unreadable; using the defaults")
            return AuthSettings()

    # -- chats --------------------------------------------------------------------

    def list_chats(self, user_id: str) -> list[dict[str, Any]]:
        """Every chat this user has, most recently updated first."""
        with self._tx() as db:
            rows = db.execute(
                "SELECT id, data FROM chats WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        chats = []
        for row in rows:
            chat = json.loads(row["data"])
            chat["id"] = row["id"]
            chats.append(chat)
        return chats

    def put_chat(self, user_id: str, chat_id: str, chat: dict[str, Any]) -> None:
        """Save a chat whole, replacing any earlier copy of it."""
        with self._tx() as db:
            self._put_chat(db, user_id, chat_id, chat, replace=True)

    def import_chats(self, user_id: str, chats: list[dict[str, Any]]) -> int:
        """Add chats this user does not have yet; returns how many were new."""
        added = 0
        with self._tx() as db:
            for chat in chats:
                added += self._put_chat(db, user_id, str(chat["id"]), chat, replace=False)
        return added

    def _put_chat(
        self,
        db: sqlite3.Connection,
        user_id: str,
        chat_id: str,
        chat: dict[str, Any],
        *,
        replace: bool,
    ) -> int:
        now = _now()
        body = {key: value for key, value in chat.items() if key not in ("id", "temporary")}
        created = _int(body.get("created_at"), now)
        updated = _int(body.get("updated_at"), now)
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        cursor = db.execute(
            f"{verb} INTO chats (user_id, id, title, data, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                user_id,
                chat_id,
                str(body.get("title") or "New Chat"),
                json.dumps(body, separators=(",", ":")),
                created,
                updated,
            ),
        )
        return cursor.rowcount

    def delete_chat(self, user_id: str, chat_id: str) -> bool:
        with self._tx() as db:
            cursor = db.execute(
                "DELETE FROM chats WHERE user_id = ? AND id = ?", (user_id, chat_id)
            )
        return cursor.rowcount > 0


def _find(db: sqlite3.Connection, user_id: str) -> sqlite3.Row | None:
    row: sqlite3.Row | None = db.execute(
        f"SELECT {USER_COLUMNS} FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    return row


def _user(row: sqlite3.Row) -> User:
    return User(
        id=row["id"],
        name=row["name"],
        email=row["email"],
        role=row["role"],
        created_at=row["created_at"],
        last_active_at=row["last_active_at"],
    )


def _int(value: Any, fallback: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    return int(value)


__all__ = [
    "AccountError",
    "AccountStore",
    "AuthSettings",
    "Role",
    "SetupCodeError",
    "User",
    "token_hash",
]
