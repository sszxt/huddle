"""The first account is the admin, so it must not go to whoever reaches the page first.

On a network where every device is trusted, the first visitor to the sign-up
page would win the admin account. The first account therefore needs a setup
code (printed in the log, shown by `huddle admin-code`) unless it is created
from the PC running Huddle. These tests pin that, from both sides.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, closing
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from typer.testing import CliRunner

from huddle.accounts.passwords import hash_password
from huddle.accounts.store import AccountStore
from huddle.app import create_app
from huddle.cli import app as cli
from huddle.config import HuddleConfig

PASSWORD = "correct horse"
CODE_SHAPE = re.compile(r"^[A-HJKMNP-Z2-9]{4}-[A-HJKMNP-Z2-9]{4}$")
LAN_DEVICE = ("192.168.0.77", 51234)
THIS_PC = ("127.0.0.1", 51234)

Connect = Callable[[tuple[str, int]], httpx.AsyncClient]


@pytest.fixture
def config(huddle_config: HuddleConfig, tmp_path: Path) -> HuddleConfig:
    huddle_config.backend.autostart = False
    huddle_config.accounts.db = tmp_path / "accounts" / "huddle.db"
    return huddle_config


@asynccontextmanager
async def serving(config: HuddleConfig) -> AsyncIterator[tuple[FastAPI, Connect]]:
    """The app, and browsers that connect from a chosen address."""
    app = create_app(config)
    opened: list[httpx.AsyncClient] = []

    def connect(address: tuple[str, int]) -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=address), base_url="http://huddle.test"
        )
        opened.append(client)
        return client

    async with app.router.lifespan_context(app):
        try:
            yield app, connect
        finally:
            for client in opened:
                await client.aclose()


def body(email: str = "admin@lan", code: str | None = None) -> dict[str, str]:
    data = {"name": "Admin", "email": email, "password": PASSWORD}
    if code is not None:
        data["setup_code"] = code
    return data


def code_of(app: FastAPI) -> str:
    code = app.state.accounts.store.setup_code()
    assert code is not None
    return str(code)


def stored_code_rows(db: Path) -> int:
    with closing(sqlite3.connect(db)) as conn:
        return int(
            conn.execute("SELECT COUNT(*) FROM settings WHERE key='setup_code'").fetchone()[0]
        )


# -- who needs the code ---------------------------------------------------------------


async def test_the_first_account_from_another_device_needs_the_code(config: HuddleConfig) -> None:
    async with serving(config) as (_, connect):
        phone = connect(LAN_DEVICE)
        settings = (await phone.get("/api/auth/config")).json()
        assert settings["first_user"] is True
        assert settings["setup_code_required"] is True

        refused = await phone.post("/api/auth/signup", json=body())
        assert refused.status_code == 403
        assert "huddle admin-code" in refused.json()["detail"]

        # Nothing was created: the page still offers the first-account flow.
        assert (await phone.get("/api/auth/config")).json()["first_user"] is True


async def test_the_first_account_from_this_pc_needs_no_code(config: HuddleConfig) -> None:
    async with serving(config) as (_, connect):
        here = connect(THIS_PC)
        assert (await here.get("/api/auth/config")).json()["setup_code_required"] is False

        response = await here.post("/api/auth/signup", json=body())
        assert response.status_code == 200
        assert response.json()["user"]["role"] == "admin"


async def test_a_wrong_code_is_refused(config: HuddleConfig) -> None:
    async with serving(config) as (_, connect):
        response = await connect(LAN_DEVICE).post("/api/auth/signup", json=body(code="AAAA-AAAA"))
        assert response.status_code == 403


async def test_the_right_code_makes_the_admin_and_then_disappears(config: HuddleConfig) -> None:
    async with serving(config) as (app, connect):
        code = code_of(app)
        assert CODE_SHAPE.match(code), code
        assert stored_code_rows(config.accounts.db) == 1

        response = await connect(LAN_DEVICE).post("/api/auth/signup", json=body(code=code))

        assert response.status_code == 200
        assert response.json()["user"]["role"] == "admin"
        assert app.state.accounts.store.setup_code() is None, "one-time: gone once used"
        assert stored_code_rows(config.accounts.db) == 0


@pytest.mark.parametrize("typed", [str.lower, lambda c: c.replace("-", " "), lambda c: f" {c} "])
async def test_the_code_is_forgiving_about_how_it_is_typed(
    config: HuddleConfig, typed: Callable[[str], str]
) -> None:
    async with serving(config) as (app, connect):
        response = await connect(LAN_DEVICE).post(
            "/api/auth/signup", json=body(code=typed(code_of(app)))
        )
        assert response.status_code == 200


async def test_a_local_first_account_also_clears_the_code(config: HuddleConfig) -> None:
    async with serving(config) as (_, connect):
        assert (await connect(THIS_PC).post("/api/auth/signup", json=body())).status_code == 200
    assert stored_code_rows(config.accounts.db) == 0


async def test_later_accounts_never_need_a_code(config: HuddleConfig) -> None:
    async with serving(config) as (_, connect):
        assert (await connect(THIS_PC).post("/api/auth/signup", json=body())).status_code == 200

        phone = connect(LAN_DEVICE)
        assert (await phone.get("/api/auth/config")).json()["setup_code_required"] is False
        joined = await phone.post("/api/auth/signup", json=body("friend@lan"))
        assert joined.status_code == 200
        assert joined.json()["user"]["role"] == "pending"


# -- guessing -------------------------------------------------------------------------


async def test_guessing_the_code_is_throttled(config: HuddleConfig) -> None:
    """Eight characters is typeable, so it must not be guessable at LAN speed."""
    async with serving(config) as (app, connect):
        phone = connect(LAN_DEVICE)
        for _ in range(10):
            assert (
                await phone.post("/api/auth/signup", json=body(code="AAAA-AAAA"))
            ).status_code == 403

        # Locked out: even the right code is refused for now.
        locked = await phone.post("/api/auth/signup", json=body(code=code_of(app)))
        assert locked.status_code == 429

        # Another device is unaffected, and so is this PC.
        other = connect(("192.168.0.99", 1))
        assert (
            await other.post("/api/auth/signup", json=body(code=code_of(app)))
        ).status_code == 200


# -- the code itself ------------------------------------------------------------------


async def test_the_code_survives_a_restart(config: HuddleConfig) -> None:
    async with serving(config) as (app, _):
        first = code_of(app)
    async with serving(config) as (app, _):
        assert code_of(app) == first


async def test_the_code_is_logged_for_whoever_runs_the_pc(
    config: HuddleConfig, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="huddle.accounts"):
        async with serving(config) as (app, _):
            code = code_of(app)
    assert any(code in record.getMessage() for record in caplog.records)
    assert any("huddle admin-code" in record.getMessage() for record in caplog.records)


async def test_no_code_is_made_or_logged_once_an_account_exists(
    config: HuddleConfig, caplog: pytest.LogCaptureFixture
) -> None:
    async with serving(config) as (_, connect):
        assert (await connect(THIS_PC).post("/api/auth/signup", json=body())).status_code == 200

    # The first start rightly announced a code; only the restart is under test.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="huddle.accounts"):
        async with serving(config) as (app, _):
            assert app.state.accounts.store.setup_code() is None
    assert not [r for r in caplog.records if "setup code" in r.getMessage()]


# -- `huddle admin-code` --------------------------------------------------------------


@pytest.fixture
def cli_config(tmp_path: Path) -> tuple[list[str], Path]:
    db = tmp_path / "cli" / "huddle.db"
    path = tmp_path / "huddle.yaml"
    path.write_text(f"accounts:\n  db: {db}\ndiscovery:\n  enabled: false\n")
    return ["admin-code", "--config", str(path)], db


def test_admin_code_prints_the_code_and_prints_it_again(
    cli_config: tuple[list[str], Path],
) -> None:
    args, _ = cli_config
    first = CliRunner().invoke(cli, args)
    second = CliRunner().invoke(cli, args)

    assert first.exit_code == 0, first.output
    assert CODE_SHAPE.match(first.output.strip()), first.output
    assert second.output == first.output, "the same code until an account exists"


async def test_admin_code_matches_the_running_node(config: HuddleConfig, tmp_path: Path) -> None:
    path = tmp_path / "huddle.yaml"
    path.write_text(f"accounts:\n  db: {config.accounts.db}\ndiscovery:\n  enabled: false\n")
    async with serving(config) as (app, _):
        shown = CliRunner().invoke(cli, ["admin-code", "--config", str(path)])
        assert shown.output.strip() == code_of(app)


def test_admin_code_says_so_once_an_account_exists(cli_config: tuple[list[str], Path]) -> None:
    args, db = cli_config
    store = AccountStore(db)
    store.open()
    store.sign_up("Admin", "admin@lan", hash_password(PASSWORD))
    store.close()

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0
    assert "already exists" in result.output
    assert not CODE_SHAPE.search(result.output)


def test_admin_code_with_accounts_turned_off(tmp_path: Path) -> None:
    path = tmp_path / "huddle.yaml"
    path.write_text("accounts:\n  enabled: false\ndiscovery:\n  enabled: false\n")
    result = CliRunner().invoke(cli, ["admin-code", "--config", str(path)])
    assert result.exit_code == 0
    assert "turned off" in result.output
