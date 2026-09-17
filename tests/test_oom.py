"""Recovering from a plan that does not actually fit.

A plan is an estimate: llama.cpp's own --fit only adjusts arguments we leave
unset, and we set -ngl and --tensor-split explicitly. So when a load runs out of
device memory anyway, the coordinator replans with more margin on the device
that failed. The fake llama-server fails with the real Vulkan error lines once a
single-node launch asks for more layers than a configured limit.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from huddle.app import create_app
from huddle.config import HuddleConfig

# One discrete card with 30 MiB free: the 12-layer test model needs ~2.1 MiB per
# layer at ctx 4096, so the first plan puts all 12 layers on it.
DEVICES = (
    "Available devices:\n"
    "  Vulkan0: NVIDIA GeForce RTX 5070 (12473 MiB, 30 MiB free)\n"
    "  Vulkan1: Intel(R) Graphics (RPL-S) (48045 MiB, 43241 MiB free)\n"
)


@pytest.fixture
def small_gpu(monkeypatch: pytest.MonkeyPatch, huddle_config: HuddleConfig) -> HuddleConfig:
    monkeypatch.setenv("HUDDLE_FAKE_DEVICES", DEVICES)
    huddle_config.backend.autostart = False
    # A small step keeps the arithmetic visible: each retry takes 8 MiB, about
    # four layers, off the device that failed.
    huddle_config.planner.oom_step_mib = 8.0
    return huddle_config


@pytest.fixture
async def http(small_gpu: HuddleConfig) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(small_gpu)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://huddle.test")
    async with app.router.lifespan_context(app), client:
        yield client


async def test_replans_after_running_out_of_memory(
    http: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The device can take 9 offloaded entries: 8 layers plus the output head.
    monkeypatch.setenv("HUDDLE_FAKE_MAX_LOCAL_LAYERS", "9")

    response = await http.post("/cluster/start")
    assert response.status_code == 200, response.text
    status = response.json()

    assert status["running"] is True
    assert status["plan"]["n_gpu_layers"] == 8, "backed off to what actually fits"
    assert status["plan"]["ngl"] == 9
    assert status["plan"]["extra_reserve_mib"] == {"Vulkan0": 8.0}

    argv = (await http.get("/agent/backend")).json()["argv"]
    assert argv[argv.index("-ngl") + 1] == "9"
    await http.post("/cluster/stop")


async def test_gives_up_with_a_useful_message(
    http: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retries are bounded, and the failure says what to change."""
    monkeypatch.setenv("HUDDLE_FAKE_MAX_LOCAL_LAYERS", "0")

    response = await http.post("/cluster/start")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "out of device memory on Vulkan0 after 3 attempts" in detail
    assert "ctx_size" in detail


async def test_other_failures_are_not_retried(
    http: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a memory failure justifies a smaller plan."""
    monkeypatch.setenv("HUDDLE_FAKE_DELAY", "0")
    response = await http.post("/cluster/start", json={"model": "absent.gguf"})
    assert response.status_code == 409
    assert "out of device memory" not in response.json()["detail"]


async def test_a_stale_failure_does_not_leak_into_the_next_start(
    small_gpu: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start that fails before launching must not report the previous output.

    Otherwise an earlier out-of-memory would make an unrelated failure look like
    one, and trigger a pointless replan.
    """
    from huddle.agent.service import BackendService
    from huddle.process import ProcessError

    monkeypatch.setenv("HUDDLE_FAKE_MAX_LOCAL_LAYERS", "0")
    backend = BackendService(small_gpu)

    with pytest.raises(ProcessError):
        await backend.start(n_gpu_layers=12)
    assert any("ErrorOutOfDeviceMemory" in line for line in backend.logs())

    with pytest.raises(ProcessError, match="model not found"):
        await backend.start("absent.gguf")
    assert backend.logs() == [], "the earlier attempt's output must be gone"
