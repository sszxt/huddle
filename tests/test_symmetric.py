"""Every node equal: two `huddle serve` nodes that find each other and share.

Each node is a real app on a real port, as on two PCs: they talk to each other
over HTTP, and the only thing faked is mDNS (an in-memory hub) and llama.cpp
(the fake binaries). So this shows the nodes agree on who serves, lend and
reclaim GPUs, pass requests to the head, and hand the cluster over — against
the fakes. Whether two real machines do the same is only shown on hardware.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from huddle.agent.service import OWNER_ID_HEADER, OWNER_NAME_HEADER
from huddle.app import create_app
from huddle.config import HuddleConfig
from huddle.coordinator.service import FORWARDED_HEADER
from huddle.state import StateStore
from tests.fakes.gguf_builder import write_gguf
from tests.fakes.memory_lan import LanHub

CHAT = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}
# A GPU too small for the whole test model, so a second node is really needed.
CONSTRAINED_DEVICES = (
    "Available devices:\n"
    "  Vulkan0: NVIDIA GeForce RTX 5070 (12473 MiB, 20 MiB free)\n"
    "  Vulkan1: Intel(R) Graphics (RPL-S) (48045 MiB, 43241 MiB free)\n"
)


@pytest.fixture(autouse=True)
def constrained_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUDDLE_FAKE_DEVICES", CONSTRAINED_DEVICES)


def claim_ports(count: int) -> list[int]:
    socks = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(count)]
    try:
        for sock in socks:
            sock.bind(("127.0.0.1", 0))
        return [int(sock.getsockname()[1]) for sock in socks]
    finally:
        for sock in socks:
            sock.close()


@dataclass
class Node:
    config: HuddleConfig
    app: Any
    http: httpx.AsyncClient

    @property
    def id(self) -> str:
        node_id: str = self.app.state.identity.id
        return node_id

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.config.api.port}"


NodeFactory = Callable[..., contextlib.AbstractAsyncContextManager[Node]]


@pytest.fixture
def make_node(
    tmp_path: Path, fake_llama_server: Path, fake_rpc_server: Path
) -> Callable[..., contextlib.AbstractAsyncContextManager[Node]]:
    """Start a node on the given hub; a context manager, so it can be stopped."""

    @contextlib.asynccontextmanager
    async def start(
        hub: LanHub,
        name: str,
        *,
        models: tuple[str, ...] = ("tiny.gguf",),
        state: str | None = None,
        **overrides: Any,
    ) -> AsyncIterator[Node]:
        import uvicorn

        api, backend, rpc = claim_ports(3)
        base = tmp_path / (state or f"{name}-{api}")
        models_dir = base / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for model in models:
            if not (models_dir / model).exists():
                write_gguf(models_dir / model, n_layers=12, n_embd=256)
        raw: dict[str, Any] = {
            "node": {"name": name, "state_dir": str(base / "state")},
            "binaries": {
                "llama_server": str(fake_llama_server),
                "rpc_server": str(fake_rpc_server),
            },
            "models": {"dir": str(models_dir)},
            "backend": {"host": "127.0.0.1", "port": backend, "autostart": False},
            # Not loopback: a node bound there cannot lend, which is the point
            # of a separate test. The server itself still binds 127.0.0.1.
            "api": {"host": "0.0.0.0", "port": api},
            "rpc": {"port": rpc},
            "discovery": {
                "enabled": True,
                "advertise": "127.0.0.1",
                "settle": 0.3,
                "poll_interval": 0.2,
            },
            "supervisor": {"watch_interval": 0.2},
        }
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)
        config = HuddleConfig.model_validate(raw)
        app = create_app(config, lan_factory=hub.lan)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=api, log_level="error"))
        task = asyncio.create_task(server.serve())
        deadline = asyncio.get_running_loop().time() + 15
        while not server.started:
            if asyncio.get_running_loop().time() > deadline:
                raise RuntimeError(f"node {name} did not start")
            await asyncio.sleep(0.05)
        http = httpx.AsyncClient(base_url=f"http://127.0.0.1:{api}", timeout=60.0)
        try:
            yield Node(config=config, app=app, http=http)
        finally:
            await http.aclose()
            server.should_exit = True
            await task

    return start


async def until(check: Callable[[], Any], *, timeout: float = 15.0, what: str = "") -> Any:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        result = check()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return result
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {what or check}")
        await asyncio.sleep(0.1)


def sees(node: Node, other: Node) -> bool:
    member = node.app.state.membership.by_id(other.id)
    return member is not None and member.alive


async def load(node: Node, model: str = "tiny.gguf", **extra: Any) -> dict[str, Any]:
    response = await node.http.post("/cluster/model", json={"model": model, **extra})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_nodes_find_each_other_by_identity_not_name(make_node: NodeFactory) -> None:
    """Two fresh installs can share a hostname; they must still be two nodes."""
    hub = LanHub()
    async with make_node(hub, "box") as a, make_node(hub, "box") as b:
        assert a.id != b.id
        await until(lambda: sees(a, b) and sees(b, a), what="the nodes to see each other")

        nodes = (await a.http.get("/cluster/nodes")).json()["nodes"]
        assert {n["id"] for n in nodes} == {a.id, b.id}
        names = [n["name"] for n in nodes]
        assert "box" in names and any(n.startswith("box-") for n in names), (
            "a clashing name is made unique for display"
        )
        assert all(n["role"] == "idle" for n in nodes), "nobody serves until asked"


async def test_loading_on_one_node_borrows_the_other(make_node: NodeFactory) -> None:
    hub = LanHub()
    async with make_node(hub, "alpha") as a, make_node(hub, "beta") as b:
        await until(lambda: sees(a, b), what="alpha to see beta")

        status = await load(a)
        assert status["running"] and status["head_node"] == "alpha"
        assert status["workers"] == ["beta"], "the model needs beta's GPU too"

        lent = (await b.http.get("/agent/rpc")).json()
        assert lent["running"] and lent["owner_id"] == a.id and lent["owner_name"] == "alpha"

        # beta answers for the cluster by asking the head.
        remote = (await b.http.get("/cluster")).json()
        assert remote["head_node"] == "alpha" and remote["running"]
        overview = (await b.http.get("/cluster/nodes")).json()
        assert overview["viewer_id"] == b.id, "the page knows which PC it is on"
        assert overview["nodes"][0]["id"] == a.id and overview["nodes"][0]["role"] == "head"
        worker = next(n for n in overview["nodes"] if n["id"] == b.id)
        assert worker["role"] == "worker" and worker["lent_to"] == "alpha"

        health = (await b.http.get("/health")).json()
        assert health["status"] == "ok" and health["head"] == "alpha"
        assert health["model"] == "tiny.gguf"


async def test_chat_from_the_other_node_still_streams(make_node: NodeFactory) -> None:
    hub = LanHub()
    async with make_node(hub, "alpha") as a, make_node(hub, "beta") as b:
        await until(lambda: sees(a, b) and sees(b, a))
        await load(a)
        await until(lambda: b.app.state.membership.head() is not None, what="beta to see a head")

        chunks: list[str] = []
        async with b.http.stream(
            "POST", "/v1/chat/completions", json={**CHAT, "stream": True}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line.removeprefix("data: "))
        assert chunks[-1] == "[DONE]"
        deltas = [json.loads(c)["choices"][0]["delta"]["content"] for c in chunks[:-1]]
        assert "".join(deltas).strip() == "huddle fake response"
        assert len(deltas) > 1, "passing through a second node must not buffer the stream"


async def test_a_forwarded_request_is_never_forwarded_again(make_node: NodeFactory) -> None:
    hub = LanHub()
    async with make_node(hub, "alpha") as a, make_node(hub, "beta") as b:
        await until(lambda: sees(a, b) and sees(b, a))
        await load(a)
        await until(lambda: b.app.state.membership.head() is not None)
        response = await b.http.post(
            "/v1/chat/completions", json=CHAT, headers={FORWARDED_HEADER: "1"}
        )
        assert response.status_code == 503, "beta serves nothing itself"


async def test_loading_on_the_other_node_hands_the_cluster_over(make_node: NodeFactory) -> None:
    hub = LanHub()
    async with make_node(hub, "alpha") as a, make_node(hub, "beta") as b:
        await until(lambda: sees(a, b) and sees(b, a))
        await load(a)
        assert StateStore(Path(a.config.node.state_dir or "")).load().resume_model

        status = await load(b)
        assert status["running"] and status["head_node"] == "beta"
        assert status["workers"] == ["alpha"], "alpha lends now, after stopping"

        alpha = a.app.state.cluster.status()
        assert not alpha.running and not alpha.desired
        # Handed over: alpha must not bring its old model back at its next boot.
        assert a.app.state.cluster.state.load().resume_model is None
        lent = (await a.http.get("/agent/rpc")).json()
        assert lent["running"] and lent["owner_id"] == b.id


async def test_models_are_listed_across_nodes_and_load_where_they_live(
    make_node: NodeFactory,
) -> None:
    hub = LanHub()
    async with (
        make_node(hub, "alpha", models=("only-on-alpha.gguf",)) as a,
        make_node(hub, "beta", models=()) as b,
    ):
        await until(lambda: sees(a, b) and sees(b, a))
        listing = (await b.http.get("/cluster/models")).json()
        assert listing["available"] == [], "beta itself has nothing"
        entry = next(e for e in listing["entries"] if e["file"] == "only-on-alpha.gguf")
        assert entry["node"] == "alpha" and entry["node_id"] == a.id and not entry["local"]

        # Loaded from beta's UI, served by the node that has the file.
        status = await load(b, "only-on-alpha.gguf", node_id=a.id)
        assert status["head_node"] == "alpha" and status["model"] == "only-on-alpha.gguf"
        assert a.app.state.cluster.status().running


async def test_a_serving_node_lends_nothing(make_node: NodeFactory) -> None:
    hub = LanHub()
    async with make_node(hub, "alpha") as a:
        await load(a)
        response = await a.http.post(
            "/agent/rpc/start", headers={OWNER_ID_HEADER: "someone", OWNER_NAME_HEADER: "x"}
        )
        assert response.status_code == 409
        assert "serving a model itself" in response.json()["detail"]


async def test_a_resumed_model_comes_back_after_a_restart(make_node: NodeFactory) -> None:
    hub = LanHub()
    async with make_node(hub, "alpha", state="alpha") as a:
        await load(a)
    # The service stopping (a reboot) is not someone deciding to stop.
    async with make_node(hub, "alpha", state="alpha") as a:
        await until(lambda: a.app.state.cluster.status().running, what="the resumed model")
        assert a.app.state.cluster.status().model == "tiny.gguf"


async def test_a_stale_resume_defers_to_a_node_already_serving(make_node: NodeFactory) -> None:
    """Both nodes think they should resume; the one already serving wins."""
    hub = LanHub()
    async with make_node(hub, "beta") as b:
        await load(b)
        async with make_node(hub, "alpha", state="alpha") as a:
            await a.app.state.cluster.stop()
        # alpha last served this model; it reboots while beta is serving.
        StateStore(Path(a.config.node.state_dir or "")).set_resume("tiny.gguf", [])
        async with make_node(hub, "alpha", state="alpha") as a:
            await until(lambda: sees(a, b), what="alpha to see beta")
            await asyncio.sleep(a.config.discovery.settle + 0.5)
            assert not a.app.state.cluster.status().desired, "alpha joins instead of fighting"
            assert a.app.state.cluster.state.load().resume_model is None


async def test_a_node_that_goes_away_is_shown_offline(make_node: NodeFactory) -> None:
    hub = LanHub()
    async with make_node(hub, "alpha") as a:
        async with make_node(hub, "beta") as b:
            await until(lambda: sees(a, b))
            beta_id = b.id
        member = await until(
            lambda: (m := a.app.state.membership.by_id(beta_id)) is not None and not m.alive and m,
            what="beta to be marked offline",
        )
        assert member.error
        nodes = (await a.http.get("/cluster/nodes")).json()["nodes"]
        assert next(n for n in nodes if n["id"] == beta_id)["role"] == "offline"


# -- leases, on one agent --------------------------------------------------------------


@pytest.fixture
async def agent(huddle_config: HuddleConfig) -> AsyncIterator[httpx.AsyncClient]:
    huddle_config.backend.autostart = False
    huddle_config.rpc.lease_timeout = 0.6
    app = create_app(huddle_config)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://huddle.test") as http,
    ):
        yield http
        await http.post("/agent/rpc/stop")


def owner(node_id: str, name: str) -> dict[str, str]:
    return {OWNER_ID_HEADER: node_id, OWNER_NAME_HEADER: name}


async def test_a_lent_worker_refuses_a_second_head(agent: httpx.AsyncClient) -> None:
    first = await agent.post("/agent/rpc/start", headers=owner("aaa", "alpha"))
    assert first.status_code == 200 and first.json()["owner_name"] == "alpha"

    again = await agent.post("/agent/rpc/start", headers=owner("aaa", "alpha"))
    assert again.status_code == 200, "the same head asking twice is harmless"
    assert again.json()["pid"] == first.json()["pid"]

    other = await agent.post("/agent/rpc/start", headers=owner("bbb", "beta"))
    assert other.status_code == 409 and "lent to alpha" in other.json()["detail"]

    anonymous = await agent.post("/agent/rpc/start")
    assert anonymous.status_code == 409 and "already running" in anonymous.json()["detail"]

    stolen = await agent.post("/agent/rpc/stop", headers=owner("bbb", "beta"))
    assert stolen.status_code == 409, "another head may not stop it"
    assert (await agent.get("/agent/rpc")).json()["running"]


async def test_a_lease_nobody_renews_frees_the_gpu(agent: httpx.AsyncClient) -> None:
    """A head that lost power never says goodbye; its worker must not wait forever."""
    await agent.post("/agent/rpc/start", headers=owner("aaa", "alpha"))
    await until(lambda: _stopped(agent), timeout=5.0, what="the unrenewed worker to be stopped")


async def test_renewing_keeps_the_lease(agent: httpx.AsyncClient) -> None:
    await agent.post("/agent/rpc/start", headers=owner("aaa", "alpha"))
    for _ in range(12):  # 1.2 s: twice the lease
        await agent.get("/agent/rpc", headers=owner("aaa", "alpha"))
        await asyncio.sleep(0.1)
    assert (await agent.get("/agent/rpc")).json()["running"]


async def _stopped(agent: httpx.AsyncClient) -> bool:
    return not (await agent.get("/agent/rpc")).json()["running"]
