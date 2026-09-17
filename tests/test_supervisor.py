"""How the supervisor paces, bounds and forgives restarts.

The failure modes here are slow ones: a fast-failing cause burning every attempt
in seconds, and a long-lived cluster slowly accumulating restarts until it stops
recovering for good.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from huddle.app import create_app
from huddle.config import HuddleConfig


@pytest.fixture
async def cluster_app(huddle_config: HuddleConfig) -> AsyncIterator[tuple[Any, httpx.AsyncClient]]:
    huddle_config.backend.autostart = False
    app = create_app(huddle_config)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://huddle.test")
    async with app.router.lifespan_context(app), client:
        yield app, client


async def test_failed_restarts_back_off(cluster_app: tuple[Any, httpx.AsyncClient]) -> None:
    """Without backoff a fast failure retries every interval and burns the limit."""
    app, _ = cluster_app
    cluster = app.state.cluster
    cluster.watch_interval = 0.1
    cluster.backoff_max = 60.0
    cluster.max_restarts = 100

    assert await cluster.start_with_retry("missing.gguf") is None
    await asyncio.sleep(1.2)

    # Attempts land at ~0.1, 0.3, 0.7 and 1.5 s; unpaced it would be ~12.
    assert 1 <= cluster.status().restarts <= 4
    await cluster.stop()


async def test_backoff_is_capped(cluster_app: tuple[Any, httpx.AsyncClient]) -> None:
    app, _ = cluster_app
    cluster = app.state.cluster
    cluster.watch_interval = 0.05
    cluster.backoff_max = 0.05
    cluster.max_restarts = 100

    assert await cluster.start_with_retry("missing.gguf") is None
    # Generous window: a slow CI runner still fits ~40 capped attempts in here,
    # where uncapped backoff would have reached only five or six.
    await asyncio.sleep(2.0)
    assert cluster.status().restarts >= 8, "the cap should keep retries frequent"
    await cluster.stop()


async def test_giving_up_stays_visible_as_degraded(
    cluster_app: tuple[Any, httpx.AsyncClient],
) -> None:
    """Exhausting the limit must not make a wanted cluster look deliberately off."""
    app, http = cluster_app
    cluster = app.state.cluster
    cluster.watch_interval = 0.05
    cluster.backoff_max = 0.05
    cluster.max_restarts = 2

    assert await cluster.start_with_retry("missing.gguf") is None
    for _ in range(100):
        await asyncio.sleep(0.05)
        if not cluster.status().supervising:
            break

    status = (await http.get("/cluster")).json()
    assert status["desired"] is True, "still wanted"
    assert status["running"] is False
    assert status["supervising"] is False, "and nobody is retrying any more"
    assert "restart limit" in status["last_failure"]

    health = (await http.get("/health")).json()
    assert health["status"] == "degraded"
    assert "restart limit" in health["detail"]


async def test_explicit_start_after_giving_up_resets_the_limit(
    cluster_app: tuple[Any, httpx.AsyncClient], huddle_config: HuddleConfig
) -> None:
    """Reviving a cluster must not leave it one failure away from giving up."""
    app, http = cluster_app
    cluster = app.state.cluster
    cluster.watch_interval = 0.05
    cluster.backoff_max = 0.05
    cluster.max_restarts = 2

    assert await cluster.start_with_retry("missing.gguf") is None
    for _ in range(100):
        await asyncio.sleep(0.05)
        if not cluster.status().supervising:
            break

    response = await http.post("/cluster/start")  # the default, which exists
    assert response.status_code == 200, response.text
    status = response.json()
    assert status["running"] is True
    assert status["restarts"] == 0
    assert status["supervising"] is True
    assert status["last_failure"] is None
    await http.post("/cluster/stop")


async def test_a_stable_run_forgives_earlier_restarts(
    cluster_app: tuple[Any, httpx.AsyncClient],
) -> None:
    """Occasional crashes over weeks must not add up to a permanent give-up."""
    app, http = cluster_app
    cluster = app.state.cluster
    cluster.watch_interval = 0.1
    cluster.stable_after = 0.8

    await http.post("/cluster/start")
    os.kill((await http.get("/agent/backend")).json()["pid"], signal.SIGKILL)

    for _ in range(100):
        await asyncio.sleep(0.1)
        status = cluster.status()
        if status.running and status.restarts == 1:
            break
    else:
        raise AssertionError("never recovered")

    for _ in range(40):
        await asyncio.sleep(0.1)
        if cluster.status().restarts == 0:
            break
    else:
        raise AssertionError("restarts were never forgiven after a stable run")

    assert cluster.status().running
    await http.post("/cluster/stop")


def test_supervisor_settings_come_from_config(huddle_config: HuddleConfig) -> None:
    huddle_config.supervisor.max_restarts = 3
    huddle_config.supervisor.backoff_max = 42.0
    app = create_app(huddle_config)
    assert app.state.cluster.max_restarts == 3
    assert app.state.cluster.backoff_max == 42.0
