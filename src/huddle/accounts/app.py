"""Sign-in and chat history routes, under ``/api``.

Modelled on Open WebUI's accounts: the first account is the admin, later
sign-ups wait as "pending" until an admin approves them, and the admin can
turn sign-ups off, choose what a new sign-up becomes, and add, change or
remove users.

A signed-in browser carries an HttpOnly cookie. Writes accept only JSON and,
when the browser says where it came from, only this page's own origin: with
the cookie's SameSite=Lax that keeps another site from acting as a signed-in
user. The page is plain HTTP, so a password crosses the LAN unencrypted —
which is why the RPC and agent ports are already firewalled to the subnet.
"""

from __future__ import annotations

import asyncio
import collections
import hmac
import logging
import re
import time
from functools import partial
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from huddle.accounts.passwords import hash_password, verify_password
from huddle.accounts.store import (
    AccountError,
    AccountStore,
    AuthSettings,
    Role,
    SetupCodeError,
    User,
)
from huddle.config import AccountsConfig

log = logging.getLogger("huddle.accounts")

COOKIE = "huddle_session"
CHAT_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
# A chat is stored whole; long conversations are big, but not this big.
MAX_CHAT_BYTES = 16 * 2**20
MAX_IMPORT_BYTES = 64 * 2**20
# Failed sign-ins allowed per address before it must wait.
MAX_FAILURES = 10
FAILURE_WINDOW = 600.0

# -- request bodies -----------------------------------------------------------------


def _email(value: str) -> str:
    value = value.strip().lower()
    local, _, domain = value.partition("@")
    if not local or not domain or " " in value:
        raise ValueError("enter a valid email address")
    return value


def _name(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("enter a name")
    return value


class SignUpBody(BaseModel):
    name: str = Field(max_length=100)
    email: str = Field(max_length=254)
    password: str = Field(min_length=8, max_length=256)
    # Only the first account, and only from another device, needs one.
    setup_code: str | None = Field(default=None, max_length=64)

    _check_name = field_validator("name")(_name)
    _check_email = field_validator("email")(_email)


class SignInBody(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)


class PasswordBody(BaseModel):
    current_password: str = Field(max_length=256)
    new_password: str = Field(min_length=8, max_length=256)


class NewUserBody(SignUpBody):
    role: Role = "user"


class UserUpdateBody(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    email: str | None = Field(default=None, max_length=254)
    password: str | None = Field(default=None, min_length=8, max_length=256)
    role: Role | None = None

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str | None) -> str | None:
        return None if value is None else _name(value)

    @field_validator("email")
    @classmethod
    def _check_email(cls, value: str | None) -> str | None:
        return None if value is None else _email(value)


class ChatBody(BaseModel):
    """A chat as the web UI keeps it. Anything else it holds is kept too."""

    model_config = ConfigDict(extra="allow")

    title: str = Field(default="", max_length=1000)
    created_at: int | None = None
    updated_at: int | None = None
    messages: list[dict[str, Any]] = []


class ImportBody(BaseModel):
    chats: list[dict[str, Any]] = Field(max_length=10_000)


class AuthConfigResponse(BaseModel):
    enabled: bool
    # False when accounts are on but the database could not be opened.
    available: bool = True
    signup_enabled: bool = False
    # No account exists yet: the next sign-up becomes the admin.
    first_user: bool = False
    # ...and this visitor is not on the PC itself, so it must show the setup code.
    setup_code_required: bool = False


class SessionResponse(BaseModel):
    user: User


# -- the service the routes share -------------------------------------------------------


class Accounts:
    """This node's accounts: the store, and who a request is signed in as."""

    def __init__(self, config: AccountsConfig) -> None:
        self.config = config
        self.store = AccountStore(config.db)
        self.available = False
        self._failures: dict[str, collections.deque[float]] = {}

    async def open(self) -> None:
        # A broken database must not stop a node serving the cluster; it
        # only costs sign-in, and the routes say why.
        try:
            await asyncio.to_thread(self.store.open)
            code = await asyncio.to_thread(self.store.setup_code)
        except Exception as exc:
            log.error("accounts: cannot open %s: %s", self.config.db, exc)
            return
        self.available = True
        if code:
            # Whoever can read this log owns the PC, which is exactly who may
            # create the first (admin) account from another device.
            log.warning(
                "accounts: no account exists yet. Create the first one on this PC, or from "
                "another device with the setup code %s (`huddle admin-code` shows it again)",
                code,
            )

    async def close(self) -> None:
        await asyncio.to_thread(self.store.close)

    async def user(self, request: Request) -> User | None:
        """Who sent ``request``, from its session cookie. Looked up once per request."""
        if hasattr(request.state, "huddle_user"):
            cached: User | None = request.state.huddle_user
            return cached
        token = request.cookies.get(COOKIE)
        user = None
        if token and self.available:
            user = await asyncio.to_thread(self.store.session_user, token)
        request.state.huddle_user = user
        return user

    # Sign-in throttling, per client address. In memory: a restart forgives.

    def throttled(self, address: str) -> bool:
        failures = self._failures.get(address)
        if not failures:
            return False
        while failures and failures[0] < time.monotonic() - FAILURE_WINDOW:
            failures.popleft()
        return len(failures) >= MAX_FAILURES

    def failed(self, address: str) -> None:
        self._failures.setdefault(address, collections.deque()).append(time.monotonic())

    def succeeded(self, address: str) -> None:
        self._failures.pop(address, None)


# -- routes -------------------------------------------------------------------------


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


LOOPBACK = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})


def _on_this_pc(request: Request) -> bool:
    """Did the request come from the PC running Huddle?

    The connection's own address, never a forwarded header, which anyone could
    forge. Behind a reverse proxy on the same PC every request looks local, so
    such a setup gets no setup-code protection; the README says so.
    """
    return _client(request) in LOOPBACK


async def _json[M: BaseModel](request: Request, model: type[M], limit: int = 2**20) -> M:
    """Parse a JSON body into ``model``, refusing anything but JSON.

    Checked by hand rather than left to FastAPI so that a plain HTML form —
    which cannot send application/json — can never reach these routes.
    """
    if "application/json" not in request.headers.get("content-type", ""):
        raise HTTPException(status_code=415, detail="send JSON")
    raw = await request.body()
    if len(raw) > limit:
        raise HTTPException(status_code=413, detail="request too large")
    try:
        return model.model_validate_json(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first.get("loc", ()))
        message = str(first.get("msg", "invalid request")).removeprefix("Value error, ")
        raise HTTPException(
            status_code=422, detail=f"{field}: {message}" if field else message
        ) from exc


def _same_origin(request: Request) -> None:
    """Refuse a write that a browser says came from another site."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(status_code=403, detail="cross-site request refused")


def _set_cookie(response: Response, request: Request, token: str, lifetime: float) -> None:
    response.set_cookie(
        COOKIE,
        token,
        max_age=int(lifetime),
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )


def build_router(accounts: Accounts | None) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["accounts"], dependencies=[Depends(_same_origin)])

    @router.get("/auth/config")
    async def auth_config(request: Request) -> AuthConfigResponse:
        """What the sign-in page should offer. Public: it is asked before signing in."""
        if accounts is None:
            return AuthConfigResponse(enabled=False)
        if not accounts.available:
            return AuthConfigResponse(enabled=True, available=False)
        settings = await asyncio.to_thread(accounts.store.settings)
        users = await asyncio.to_thread(accounts.store.count_users)
        return AuthConfigResponse(
            enabled=True,
            signup_enabled=settings.signup_enabled or users == 0,
            first_user=users == 0,
            setup_code_required=users == 0 and not _on_this_pc(request),
        )

    if accounts is None:
        return router
    store = accounts.store
    lifetime = accounts.config.session_days * 86400

    async def signed_in(request: Request) -> User:
        user = await accounts.user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="sign in first")
        return user

    async def active(request: Request) -> User:
        user = await signed_in(request)
        if not user.active:
            raise HTTPException(
                status_code=403, detail="your account is waiting for an admin to approve it"
            )
        return user

    async def admin(request: Request) -> User:
        user = await signed_in(request)
        if user.role != "admin":
            raise HTTPException(status_code=403, detail="only an admin can do that")
        return user

    async def run(func: Any, *args: Any, **kwargs: Any) -> Any:
        """A store call off the event loop, its refusals as HTTP errors."""
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except AccountError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    # -- signing in -----------------------------------------------------------------

    @router.post("/auth/signup")
    async def sign_up(request: Request, response: Response) -> SessionResponse:
        address = _client(request)
        if accounts.throttled(address):
            raise HTTPException(
                status_code=429, detail="too many failed attempts; try again in a few minutes"
            )
        body = await _json(request, SignUpBody)
        password_hash = await asyncio.to_thread(hash_password, body.password)
        try:
            user: User = await asyncio.to_thread(
                partial(
                    store.sign_up,
                    body.name,
                    body.email,
                    password_hash,
                    setup_code=body.setup_code,
                    trusted=_on_this_pc(request),
                )
            )
        except SetupCodeError as exc:
            # Counts against the same limit as a wrong password: the code is
            # short enough to type, so it must not be guessable at LAN speed.
            accounts.failed(address)
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        except AccountError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        log.info("accounts: %s signed up as %s", user.email, user.role)
        _set_cookie(response, request, await run(store.create_session, user.id, lifetime), lifetime)
        return SessionResponse(user=user)

    @router.post("/auth/signin")
    async def sign_in(request: Request, response: Response) -> SessionResponse:
        address = _client(request)
        if accounts.throttled(address):
            raise HTTPException(
                status_code=429, detail="too many failed sign-ins; try again in a few minutes"
            )
        body = await _json(request, SignInBody)
        record = await run(store.login_record, body.email.strip().lower())
        # Hash even for an unknown email, so timing does not say which exist.
        stored = record[1] if record else "scrypt$16384$8$1$AAAAAAAAAAAAAAAAAAAAAA==$AA=="
        valid = await asyncio.to_thread(verify_password, body.password, stored)
        if record is None or not valid:
            accounts.failed(address)
            raise HTTPException(status_code=401, detail="the email or password is wrong")
        accounts.succeeded(address)
        user = record[0]
        _set_cookie(response, request, await run(store.create_session, user.id, lifetime), lifetime)
        return SessionResponse(user=user)

    @router.post("/auth/signout")
    async def sign_out(request: Request, response: Response) -> dict[str, bool]:
        token = request.cookies.get(COOKIE)
        if token:
            await run(store.delete_session, token)
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @router.get("/auth/me")
    async def me(request: Request) -> SessionResponse:
        user = await signed_in(request)
        await run(store.touch, user.id)
        return SessionResponse(user=user)

    @router.post("/auth/password")
    async def change_password(request: Request, response: Response) -> dict[str, bool]:
        user = await signed_in(request)
        body = await _json(request, PasswordBody)
        stored = await run(store.password_hash, user.id)
        if stored is None or not await asyncio.to_thread(
            verify_password, body.current_password, stored
        ):
            raise HTTPException(status_code=403, detail="the current password is wrong")
        password_hash = await asyncio.to_thread(hash_password, body.new_password)
        # A new password signs the account out everywhere; keep this browser in.
        await run(store.update_user, user.id, password_hash=password_hash)
        _set_cookie(response, request, await run(store.create_session, user.id, lifetime), lifetime)
        return {"ok": True}

    # -- chat history ---------------------------------------------------------------

    @router.get("/chats")
    async def list_chats(request: Request) -> list[dict[str, Any]]:
        user = await active(request)
        chats: list[dict[str, Any]] = await run(store.list_chats, user.id)
        return chats

    @router.put("/chats/{chat_id}")
    async def save_chat(chat_id: str, request: Request) -> dict[str, bool]:
        user = await active(request)
        if not CHAT_ID.match(chat_id):
            raise HTTPException(status_code=422, detail="invalid chat id")
        body = await _json(request, ChatBody, limit=MAX_CHAT_BYTES)
        await run(store.put_chat, user.id, chat_id, body.model_dump())
        return {"ok": True}

    @router.delete("/chats/{chat_id}")
    async def delete_chat(chat_id: str, request: Request) -> dict[str, bool]:
        user = await active(request)
        if not await run(store.delete_chat, user.id, chat_id):
            raise HTTPException(status_code=404, detail="no such chat")
        return {"ok": True}

    @router.post("/chats/import")
    async def import_chats(request: Request) -> dict[str, int]:
        """Take in the chats a browser kept before accounts existed."""
        user = await active(request)
        body = await _json(request, ImportBody, limit=MAX_IMPORT_BYTES)
        chats = []
        for raw in body.chats:
            chat_id = str(raw.get("id", ""))
            if not CHAT_ID.match(chat_id) or raw.get("temporary"):
                continue
            try:
                chat = ChatBody.model_validate(raw).model_dump()
            except ValidationError:
                continue
            chats.append({**chat, "id": chat_id})
        imported: int = await run(store.import_chats, user.id, chats)
        return {"imported": imported, "skipped": len(body.chats) - imported}

    # -- administration -------------------------------------------------------------

    @router.get("/users")
    async def list_users(request: Request) -> list[User]:
        await admin(request)
        users: list[User] = await run(store.list_users)
        return users

    @router.post("/users")
    async def add_user(request: Request) -> User:
        await admin(request)
        body = await _json(request, NewUserBody)
        password_hash = await asyncio.to_thread(hash_password, body.password)
        user: User = await run(store.add_user, body.name, body.email, password_hash, body.role)
        return user

    @router.patch("/users/{user_id}")
    async def update_user(user_id: str, request: Request) -> User:
        await admin(request)
        body = await _json(request, UserUpdateBody)
        password_hash = (
            await asyncio.to_thread(hash_password, body.password) if body.password else None
        )
        user: User = await run(
            store.update_user,
            user_id,
            name=body.name,
            email=body.email,
            role=body.role,
            password_hash=password_hash,
        )
        return user

    @router.delete("/users/{user_id}")
    async def delete_user(user_id: str, request: Request) -> dict[str, bool]:
        await admin(request)
        await run(store.delete_user, user_id)
        return {"ok": True}

    @router.get("/settings/auth")
    async def get_settings(request: Request) -> AuthSettings:
        await admin(request)
        settings: AuthSettings = await run(store.settings)
        return settings

    @router.put("/settings/auth")
    async def put_settings(request: Request) -> AuthSettings:
        await admin(request)
        body = await _json(request, AuthSettings)
        saved: AuthSettings = await run(store.save_settings, body)
        return saved

    return router


def bearer_matches(request: Request, expected: str) -> bool:
    token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    return hmac.compare_digest(token.encode(), expected.encode())


__all__ = ["COOKIE", "Accounts", "bearer_matches", "build_router"]
