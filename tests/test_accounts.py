"""Web UI accounts: sign-up and approval, sessions, and per-user chat history.

Driven through the real app, with its SQLite file in the test's own directory.
No model is loaded: none of this needs one.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import Any

import httpx
import pytest

from huddle.accounts.passwords import hash_password, verify_password
from huddle.accounts.store import AccountStore
from huddle.app import create_app
from huddle.config import HuddleConfig

PASSWORD = "correct horse"
Clients = Callable[[], httpx.AsyncClient]


@pytest.fixture
def config(huddle_config: HuddleConfig, tmp_path: Path) -> HuddleConfig:
    huddle_config.backend.autostart = False
    huddle_config.accounts.db = tmp_path / "accounts" / "huddle.db"
    return huddle_config


@asynccontextmanager
async def running(config: HuddleConfig) -> AsyncIterator[Clients]:
    """The app, and a way to open browsers against it: each keeps its own cookies."""
    app = create_app(config)
    opened: list[httpx.AsyncClient] = []

    def browser() -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://huddle.test"
        )
        opened.append(client)
        return client

    async with app.router.lifespan_context(app):
        try:
            yield browser
        finally:
            for client in opened:
                await client.aclose()


@pytest.fixture
async def browsers(config: HuddleConfig) -> AsyncIterator[Clients]:
    async with running(config) as browser:
        yield browser


async def sign_up(
    client: httpx.AsyncClient, email: str, name: str = "Someone", password: str = PASSWORD
) -> httpx.Response:
    return await client.post(
        "/api/auth/signup", json={"name": name, "email": email, "password": password}
    )


async def sign_in(
    client: httpx.AsyncClient, email: str, password: str = PASSWORD
) -> httpx.Response:
    return await client.post("/api/auth/signin", json={"email": email, "password": password})


async def admin_and_user(browser: Clients) -> tuple[httpx.AsyncClient, httpx.AsyncClient]:
    """An admin and an approved user, each signed in in their own browser."""
    admin, user = browser(), browser()
    assert (await sign_up(admin, "admin@lan", "Admin")).status_code == 200
    joined = (await sign_up(user, "user@lan", "User")).json()["user"]
    await admin.patch(f"/api/users/{joined['id']}", json={"role": "user"})
    assert (await sign_in(user, "user@lan")).status_code == 200
    return admin, user


def chat(chat_id: str, updated_at: int, title: str = "hello") -> dict[str, Any]:
    return {
        "id": chat_id,
        "title": title,
        "created_at": 1,
        "updated_at": updated_at,
        "params": {"system": "be brief"},
        "messages": [{"id": "m1", "role": "user", "content": "hi"}],
    }


# -- passwords ------------------------------------------------------------------------


def test_password_hash_round_trip() -> None:
    stored = hash_password(PASSWORD)
    assert PASSWORD not in stored
    assert verify_password(PASSWORD, stored)
    assert not verify_password("wrong", stored)
    assert hash_password(PASSWORD) != stored, "each hash must get its own salt"


def test_a_malformed_hash_never_verifies() -> None:
    assert not verify_password(PASSWORD, "")
    assert not verify_password(PASSWORD, "bcrypt$whatever")
    assert not verify_password(PASSWORD, "scrypt$x$8$1$AAAA$AAAA")


# -- signing up -----------------------------------------------------------------------


async def test_the_first_account_is_the_admin(browsers: Clients) -> None:
    client = browsers()
    config = (await client.get("/api/auth/config")).json()
    assert config == {
        "enabled": True,
        "available": True,
        "signup_enabled": True,
        "first_user": True,
        # httpx's ASGI transport connects from 127.0.0.1: the PC itself.
        "setup_code_required": False,
    }

    response = await sign_up(client, "Admin@LAN ", "Admin")
    assert response.status_code == 200
    assert response.json()["user"]["role"] == "admin"
    assert response.json()["user"]["email"] == "admin@lan", "emails are kept trimmed, lower case"

    me = (await client.get("/api/auth/me")).json()["user"]
    assert me["name"] == "Admin"
    assert (await client.get("/api/auth/config")).json()["first_user"] is False


async def test_later_sign_ups_wait_for_approval(browsers: Clients) -> None:
    admin, newcomer = browsers(), browsers()
    await sign_up(admin, "admin@lan")
    joined = await sign_up(newcomer, "new@lan")
    assert joined.json()["user"]["role"] == "pending"

    # Signed in, but only to be told to wait.
    assert (await newcomer.get("/api/auth/me")).json()["user"]["role"] == "pending"
    waiting = await newcomer.get("/api/chats")
    assert waiting.status_code == 403
    assert "approve" in waiting.json()["detail"]

    user_id = joined.json()["user"]["id"]
    approved = await admin.patch(f"/api/users/{user_id}", json={"role": "user"})
    assert approved.json()["role"] == "user"
    assert (await newcomer.get("/api/chats")).status_code == 200


async def test_an_admin_can_let_sign_ups_in_directly_or_turn_them_off(browsers: Clients) -> None:
    admin = browsers()
    await sign_up(admin, "admin@lan")

    await admin.put("/api/settings/auth", json={"signup_enabled": True, "default_role": "user"})
    direct = await sign_up(browsers(), "direct@lan")
    assert direct.json()["user"]["role"] == "user"

    await admin.put("/api/settings/auth", json={"signup_enabled": False, "default_role": "user"})
    assert (await browsers().get("/api/auth/config")).json()["signup_enabled"] is False
    refused = await sign_up(browsers(), "late@lan")
    assert refused.status_code == 403
    assert "turned off" in refused.json()["detail"]


async def test_sign_up_validates_its_fields(browsers: Clients) -> None:
    client = browsers()
    short = await sign_up(client, "a@lan", password="short")
    assert short.status_code == 422
    assert "password" in short.json()["detail"]
    assert (await sign_up(client, "not-an-email")).status_code == 422
    assert (await sign_up(client, "a@lan", name="   ")).status_code == 422

    await sign_up(client, "taken@lan")
    again = await sign_up(browsers(), "TAKEN@lan")
    assert again.status_code == 409, "emails are unique regardless of case"


# -- signing in -----------------------------------------------------------------------


async def test_sign_in_and_out(browsers: Clients) -> None:
    await sign_up(browsers(), "admin@lan")
    client = browsers()
    assert (await client.get("/api/auth/me")).status_code == 401

    wrong = await sign_in(client, "admin@lan", "wrong password")
    assert wrong.status_code == 401
    unknown = await sign_in(client, "nobody@lan")
    assert unknown.json()["detail"] == wrong.json()["detail"], "must not say which emails exist"

    assert (await sign_in(client, " ADMIN@lan")).status_code == 200
    assert (await client.get("/api/auth/me")).status_code == 200

    await client.post("/api/auth/signout")
    assert (await client.get("/api/auth/me")).status_code == 401


async def test_repeated_failures_are_throttled(browsers: Clients) -> None:
    await sign_up(browsers(), "admin@lan")
    client = browsers()
    for _ in range(10):
        assert (await sign_in(client, "admin@lan", "guess")).status_code == 401
    blocked = await sign_in(client, "admin@lan")
    assert blocked.status_code == 429, "even the right password waits once throttled"


async def test_changing_a_password_signs_out_other_browsers(browsers: Clients) -> None:
    here, elsewhere = browsers(), browsers()
    await sign_up(here, "admin@lan")
    await sign_in(elsewhere, "admin@lan")

    wrong = await here.post(
        "/api/auth/password", json={"current_password": "nope", "new_password": "new password"}
    )
    assert wrong.status_code == 403

    changed = await here.post(
        "/api/auth/password",
        json={"current_password": PASSWORD, "new_password": "new password"},
    )
    assert changed.status_code == 200
    assert (await here.get("/api/auth/me")).status_code == 200, "this browser stays signed in"
    assert (await elsewhere.get("/api/auth/me")).status_code == 401
    assert (await sign_in(browsers(), "admin@lan", "new password")).status_code == 200


async def test_sessions_survive_a_restart(config: HuddleConfig) -> None:
    async with running(config) as browser:
        client = browser()
        await sign_up(client, "admin@lan")
        cookies = dict(client.cookies)
    async with running(config) as browser:
        client = browser()
        client.cookies.update(cookies)
        assert (await client.get("/api/auth/me")).json()["user"]["email"] == "admin@lan"


# -- chat history ---------------------------------------------------------------------


async def test_chats_are_saved_listed_and_deleted(browsers: Clients) -> None:
    client = browsers()
    await sign_up(client, "admin@lan")
    assert (await client.get("/api/chats")).json() == []

    assert (await client.put("/api/chats/older", json=chat("older", 10))).status_code == 200
    assert (await client.put("/api/chats/newer", json=chat("newer", 20))).status_code == 200
    listed = (await client.get("/api/chats")).json()
    assert [c["id"] for c in listed] == ["newer", "older"], "most recent first"
    assert listed[0]["params"] == {"system": "be brief"}, "everything the UI keeps comes back"
    assert listed[0]["messages"][0]["content"] == "hi"

    await client.put("/api/chats/older", json=chat("older", 30, title="renamed"))
    listed = (await client.get("/api/chats")).json()
    assert [(c["id"], c["title"]) for c in listed] == [("older", "renamed"), ("newer", "hello")]

    assert (await client.delete("/api/chats/older")).status_code == 200
    assert (await client.delete("/api/chats/older")).status_code == 404
    assert [c["id"] for c in (await client.get("/api/chats")).json()] == ["newer"]


async def test_each_user_sees_only_their_own_chats(browsers: Clients) -> None:
    admin, user = await admin_and_user(browsers)
    await admin.put("/api/chats/same-id", json=chat("same-id", 1, title="admin's"))
    await user.put("/api/chats/same-id", json=chat("same-id", 1, title="user's"))

    assert [c["title"] for c in (await admin.get("/api/chats")).json()] == ["admin's"]
    assert [c["title"] for c in (await user.get("/api/chats")).json()] == ["user's"]
    await user.delete("/api/chats/same-id")
    assert len((await admin.get("/api/chats")).json()) == 1, "deleting mine left theirs"


async def test_chats_need_a_signed_in_user(browsers: Clients) -> None:
    await sign_up(browsers(), "admin@lan")
    stranger = browsers()
    assert (await stranger.get("/api/chats")).status_code == 401
    assert (await stranger.put("/api/chats/x", json=chat("x", 1))).status_code == 401


async def test_a_bad_chat_id_is_refused(browsers: Clients) -> None:
    client = browsers()
    await sign_up(client, "admin@lan")
    assert (await client.put("/api/chats/has%20space", json=chat("x", 1))).status_code == 422


async def test_importing_a_browsers_chats(browsers: Clients) -> None:
    client = browsers()
    await sign_up(client, "admin@lan")
    await client.put("/api/chats/kept", json=chat("kept", 5, title="server copy"))

    temporary = {**chat("temp", 1), "temporary": True}
    response = await client.post(
        "/api/chats/import",
        json={"chats": [chat("kept", 9, title="browser copy"), chat("fresh", 3), temporary]},
    )
    assert response.json() == {"imported": 1, "skipped": 2}
    titles = {c["id"]: c["title"] for c in (await client.get("/api/chats")).json()}
    assert titles == {"kept": "server copy", "fresh": "hello"}, "existing chats are kept"


# -- administration -------------------------------------------------------------------


async def test_only_an_admin_manages_users(browsers: Clients) -> None:
    admin, user = await admin_and_user(browsers)
    assert (await user.get("/api/users")).status_code == 403
    assert (await user.put("/api/settings/auth", json={"signup_enabled": False})).status_code == 403

    users = (await admin.get("/api/users")).json()
    assert [u["email"] for u in users] == ["admin@lan", "user@lan"]
    assert all("password" not in key for u in users for key in u)


async def test_an_admin_adds_edits_and_removes_users(browsers: Clients) -> None:
    admin = browsers()
    await sign_up(admin, "admin@lan")
    added = await admin.post(
        "/api/users",
        json={"name": "Guest", "email": "guest@lan", "password": PASSWORD, "role": "user"},
    )
    assert added.status_code == 200
    guest_id = added.json()["id"]

    guest = browsers()
    assert (await sign_in(guest, "guest@lan")).status_code == 200
    await guest.put("/api/chats/g1", json=chat("g1", 1))

    reset = await admin.patch(f"/api/users/{guest_id}", json={"password": "reset password"})
    assert reset.status_code == 200
    assert (await guest.get("/api/chats")).status_code == 401, "a reset signs the user out"
    assert (await sign_in(guest, "guest@lan", "reset password")).status_code == 200

    assert (await admin.delete(f"/api/users/{guest_id}")).status_code == 200
    assert (await guest.get("/api/auth/me")).status_code == 401
    assert (await sign_in(browsers(), "guest@lan", "reset password")).status_code == 401


async def test_the_last_admin_cannot_be_removed(browsers: Clients) -> None:
    admin = browsers()
    me = (await sign_up(admin, "admin@lan")).json()["user"]
    demoted = await admin.patch(f"/api/users/{me['id']}", json={"role": "user"})
    assert demoted.status_code == 409
    assert (await admin.delete(f"/api/users/{me['id']}")).status_code == 409

    second = await admin.post(
        "/api/users",
        json={"name": "Two", "email": "two@lan", "password": PASSWORD, "role": "admin"},
    )
    assert (await admin.patch(f"/api/users/{me['id']}", json={"role": "user"})).status_code == 200
    assert second.json()["role"] == "admin"


# -- request hygiene ------------------------------------------------------------------


async def test_writes_from_another_site_are_refused(browsers: Clients) -> None:
    client = browsers()
    await sign_up(client, "admin@lan")
    forged = await client.put(
        "/api/chats/x", json=chat("x", 1), headers={"origin": "http://evil.example"}
    )
    assert forged.status_code == 403
    same = await client.put(
        "/api/chats/x", json=chat("x", 1), headers={"origin": "http://huddle.test"}
    )
    assert same.status_code == 200


async def test_writes_must_be_json(browsers: Clients) -> None:
    client = browsers()
    form = await client.post(
        "/api/auth/signup",
        content=b"name=a&email=a@lan&password=aaaaaaaa",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert form.status_code == 415


async def test_the_database_holds_no_passwords_or_session_tokens(
    browsers: Clients, config: HuddleConfig
) -> None:
    client = browsers()
    await sign_up(client, "admin@lan")
    token = client.cookies["huddle_session"]
    with closing(sqlite3.connect(config.accounts.db)) as db:
        dump = "\n".join(db.iterdump())
    assert PASSWORD not in dump
    assert token not in dump


# -- the OpenAI API ----------------------------------------------------------------------


async def test_a_signed_in_user_needs_no_api_key(config: HuddleConfig) -> None:
    config.api.api_key = "s3cret"
    async with running(config) as browser:
        stranger, admin, pending = browser(), browser(), browser()
        await sign_up(admin, "admin@lan")
        await sign_up(pending, "pending@lan")

        # No model is loaded, so passing the key check ends at 503, not 200.
        assert (await stranger.get("/v1/models")).status_code == 401
        assert (await pending.get("/v1/models")).status_code == 401
        assert (await admin.get("/v1/models")).status_code == 503
        keyed = await stranger.get("/v1/models", headers={"authorization": "Bearer s3cret"})
        assert keyed.status_code == 503


async def test_accounts_can_be_turned_off(config: HuddleConfig) -> None:
    config.accounts.enabled = False
    async with running(config) as browser:
        client = browser()
        assert (await client.get("/api/auth/config")).json()["enabled"] is False
        assert (await client.get("/api/chats")).status_code == 404
    assert not config.accounts.db.exists(), "no database when accounts are off"


async def test_a_database_that_will_not_open_leaves_the_node_running(
    config: HuddleConfig,
) -> None:
    config.accounts.db.parent.mkdir(parents=True)
    config.accounts.db.mkdir()  # a directory where the file should be
    async with running(config) as browser:
        client = browser()
        assert (await client.get("/api/auth/config")).json()["available"] is False
        assert (await sign_up(client, "admin@lan")).status_code == 503
        assert (await client.get("/health")).status_code == 200


def test_a_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "huddle.db"
    store = AccountStore(path)
    store.open()
    store.close()
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(RuntimeError, match="newer Huddle"):
        AccountStore(path).open()
