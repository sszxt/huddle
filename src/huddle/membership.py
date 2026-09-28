"""Who is in the cluster, and what each of them is doing right now.

Two sources, each used for what it is good at. mDNS finds nodes: a record
appears within a second of a node starting and carries where to dial it. It is
bad at saying a node is *gone* — records outlive a node that lost power by an
hour — so liveness and state come from asking each node directly, every few
seconds, over ``GET /agent/state``. That answer is cheap by contract: no
hardware probe, no subprocess.

Every node keeps its own table; there is no leader. A node decides what to do
from what the others say they are doing: whoever is serving a model is the
head, and a node lending its GPU says to whom.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ValidationError

from huddle.config import HuddleConfig, PeerConfig
from huddle.discovery import PROTO, DiscoveredPeer, Identity, OnDown, OnUp
from huddle.llamacpp import same_build

log = logging.getLogger("huddle.membership")

Role = Literal["idle", "head", "worker"]
Source = Literal["configured", "discovered", "remembered"]

# Asking a node what it is doing. A live node on a LAN answers in milliseconds.
POLL_TIMEOUT = httpx.Timeout(2.0, connect=1.0)
# Consecutive unanswered polls before a node counts as offline, so one slow
# answer does not make a node flicker off the cluster page.
MISSES_OFFLINE = 2


class NodeReport(BaseModel):
    """``GET /agent/state``: what a node is doing, cheaply."""

    id: str | None = None
    name: str
    cluster: str = "default"
    proto: int = PROTO
    huddle: str | None = None
    llamacpp: str | None = None
    # "head" from the moment a node is asked to serve a model — while loading,
    # and while its supervisor retries — until it is stopped.
    role: Role = "idle"
    # The model loaded, or being loaded when `starting`.
    model: str | None = None
    starting: bool = False
    # Who this node's GPU is lent to, while it runs a worker for someone.
    owner_id: str | None = None
    owner_name: str | None = None
    # The peers a head is using.
    workers: list[str] = []
    # False for `huddle agent`, which lends but never serves.
    can_head: bool = True
    resume_since: float | None = None


class Lan(Protocol):
    """Where nodes announce themselves. ``ZeroconfLan`` for real, a hub in tests."""

    async def start(self, on_up: OnUp, on_down: OnDown) -> None: ...

    async def stop(self) -> None: ...


@dataclass
class Member:
    """Another node, as this one currently knows it."""

    key: str
    name: str
    host: str
    port: int
    rpc_port: int
    source: Source
    id: str | None = None
    llamacpp: str | None = None
    service: str | None = None
    report: NodeReport | None = None
    misses: int = 0
    error: str | None = None
    # `name`, made unique when two nodes share one.
    display: str = ""

    @property
    def alive(self) -> bool:
        return self.report is not None and self.misses < MISSES_OFFLINE

    @property
    def is_head(self) -> bool:
        return self.alive and self.report is not None and self.report.role == "head"

    def peer(self) -> PeerConfig:
        return PeerConfig(
            name=self.display or self.name,
            host=self.host,
            agent_port=self.port,
            rpc_port=self.rpc_port,
            id=self.id,
        )


def _describe(exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectTimeout):
        return "no answer"
    if isinstance(exc, httpx.ConnectError):
        return "not reachable"
    if isinstance(exc, httpx.TimeoutException):
        return "too slow to answer"
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


class Membership:
    """This node's live view of every other node."""

    def __init__(
        self,
        config: HuddleConfig,
        identity: Identity,
        lan: Lan | None = None,
        *,
        remembered: Iterable[PeerConfig] = (),
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.identity = identity
        self.lan = lan
        self.members: dict[str, Member] = {}
        # Set whenever a node appears, disappears or changes role. Consumers
        # clear it once they have acted on it.
        self.changed = asyncio.Event()
        self._client = client
        self._own_client = client is None
        self._poller: asyncio.Task[None] | None = None
        self._settled_at: float | None = None
        self._polls: set[asyncio.Task[None]] = set()

        for peer in config.peers:
            self.members[f"cfg:{peer.name}"] = Member(
                key=f"cfg:{peer.name}",
                name=peer.name,
                host=peer.host,
                port=peer.agent_port,
                rpc_port=peer.rpc_port,
                source="configured",
                id=peer.id,
            )
        for peer in remembered:
            if peer.id == identity.id or self._configured_match(peer.name, peer.id):
                continue
            key = peer.id or f"name:{peer.name}"
            self.members[key] = Member(
                key=key,
                name=peer.name,
                host=peer.host,
                port=peer.agent_port,
                rpc_port=peer.rpc_port,
                source="remembered",
                id=peer.id,
                # Not known to be up until it answers.
                misses=MISSES_OFFLINE,
            )
        self._assign_display_names()

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._settled_at = loop.time() + self.config.discovery.settle
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=POLL_TIMEOUT)
        if self.lan is not None:
            await self.lan.start(self._on_up, self._on_down)
        self._poller = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        for task in [self._poller, *self._polls]:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._poller = None
        if self.lan is not None:
            await self.lan.stop()
        if self._own_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def wait_settled(self) -> None:
        """Wait out the startup listening window, then ask everyone once."""
        if self._settled_at is not None:
            remaining = self._settled_at - asyncio.get_running_loop().time()
            if remaining > 0:
                await asyncio.sleep(remaining)
        await self.refresh()

    # -- what mDNS says -----------------------------------------------------------

    def _configured_match(self, name: str, node_id: str | None) -> Member | None:
        for member in self.members.values():
            if member.source != "configured":
                continue
            if member.name == name or (node_id is not None and member.id == node_id):
                return member
        return None

    def _on_up(self, peer: DiscoveredPeer) -> None:
        if peer.id is not None and peer.id == self.identity.id:
            return
        if peer.id is None and peer.name == self.identity.name:
            return  # an old record of ours, from before identities
        if (peer.cluster or "default") != self.config.discovery.cluster:
            return
        if peer.is_coordinator:
            # A loopback-only `huddle serve`: its agent routes are not on the
            # network, so it cannot lend, and nothing can be forwarded to it.
            return

        configured = self._configured_match(peer.name, peer.id)
        if configured is not None:
            # Writing an address down is a decision; discovery only fills in
            # the identity it could not have known.
            configured.id = configured.id or peer.id
            configured.llamacpp = configured.llamacpp or peer.llamacpp_version
            return

        member = self.members.get(peer.key)
        if member is None:
            member = Member(
                key=peer.key,
                name=peer.name,
                host=peer.host,
                port=peer.agent_port,
                rpc_port=peer.rpc_port,
                source="discovered",
                id=peer.id,
            )
            self.members[peer.key] = member
            log.info("membership: found %s at %s:%d", peer.name, peer.host, peer.agent_port)
            self.changed.set()
        elif (member.host, member.port) != (peer.host, peer.agent_port):
            log.info("membership: %s moved to %s:%d", peer.name, peer.host, peer.agent_port)
            self.changed.set()
        member.name, member.host = peer.name, peer.host
        member.port, member.rpc_port = peer.agent_port, peer.rpc_port
        member.source, member.service = "discovered", peer.service
        member.llamacpp = peer.llamacpp_version
        self._assign_display_names()
        self._poll_soon(member)

    def _on_down(self, service: str) -> None:
        for member in self.members.values():
            if member.service == service and member.alive:
                # A goodbye: the node stopped cleanly. It stays listed, as
                # offline, so the page can say where it went.
                log.info("membership: %s left", member.name)
                member.misses, member.error = MISSES_OFFLINE, "stopped"
                self.changed.set()

    # -- what nodes say about themselves --------------------------------------------

    def _poll_soon(self, member: Member) -> None:
        task = asyncio.create_task(self._poll(member))
        self._polls.add(task)
        task.add_done_callback(self._polls.discard)

    async def _poll_loop(self) -> None:
        while True:
            await self.refresh()
            await asyncio.sleep(self.config.discovery.poll_interval)

    async def refresh(self, key: str | None = None) -> None:
        """Ask one node (by key), or every node, what it is doing now."""
        if key is not None:
            member = self.members.get(key)
            if member is not None:
                await self._poll(member)
            return
        await asyncio.gather(*(self._poll(member) for member in list(self.members.values())))

    async def _poll(self, member: Member) -> None:
        client = self._client
        if client is None:
            return
        base = f"http://{member.host}:{member.port}"
        was_alive, was_role = member.alive, member.report.role if member.report else None
        try:
            response = await client.get(f"{base}/agent/state", timeout=POLL_TIMEOUT)
            if response.status_code == 404:
                report = await self._legacy_report(client, base, member)
            else:
                response.raise_for_status()
                report = NodeReport.model_validate(response.json())
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            member.misses += 1
            member.error = _describe(exc)
            if was_alive and not member.alive:
                log.info("membership: %s is offline (%s)", member.name, member.error)
                self.changed.set()
            return

        if report.id is not None and report.id == self.identity.id:
            # A configured or remembered address that turned out to be us.
            self.members.pop(member.key, None)
            return
        if report.id is not None:
            member.id = member.id or report.id
        member.report, member.misses, member.error = report, 0, None
        member.llamacpp = report.llamacpp or member.llamacpp
        if not was_alive or was_role != report.role:
            self.changed.set()

    async def _legacy_report(
        self, client: httpx.AsyncClient, base: str, member: Member
    ) -> NodeReport:
        """An agent from before /agent/state: all it can do is lend."""
        response = await client.get(f"{base}/agent/rpc", timeout=POLL_TIMEOUT)
        response.raise_for_status()
        rpc = response.json()
        busy = bool(rpc.get("running") or rpc.get("foreign"))
        return NodeReport(
            name=member.name,
            cluster=self.config.discovery.cluster,
            llamacpp=member.llamacpp,
            role="worker" if busy else "idle",
            can_head=False,
        )

    # -- answers ------------------------------------------------------------------

    def _assign_display_names(self) -> None:
        counts = Counter(member.name for member in self.members.values())
        counts[self.identity.name] += 1
        for member in self.members.values():
            suffix = (member.id or member.host).replace(".", "")[:4]
            member.display = member.name if counts[member.name] == 1 else f"{member.name}-{suffix}"

    def known(self) -> list[Member]:
        """Every other node, configured ones first, then by name."""
        order = {"configured": 0, "discovered": 1, "remembered": 1}
        return sorted(self.members.values(), key=lambda m: (order[m.source], m.display))

    def by_id(self, node_id: str) -> Member | None:
        return next((m for m in self.members.values() if m.id == node_id), None)

    def heads(self) -> list[Member]:
        return [member for member in self.members.values() if member.is_head]

    def head(self) -> Member | None:
        """The other node serving a model, if any."""
        heads = self.heads()
        # Two heads can briefly coexist during a handover; the one serving
        # already beats the one still loading.
        heads.sort(key=lambda m: (m.report is not None and m.report.starting, m.display))
        return heads[0] if heads else None

    def plan_peers(self, local_version: str | None) -> list[PeerConfig]:
        """The peers a plan made now should consider, best first.

        Configured peers always (an unreachable one is reported by the plan);
        live nodes that are free to lend; and nodes this one last ran with that
        have not answered yet, so the plan counts them as unreachable and the
        start waits for them rather than shrinking the cluster.
        """
        peers: list[PeerConfig] = []
        for member in self.known():
            if member.source == "configured":
                peers.append(member.peer())
                continue
            if not member.alive:
                if member.source == "remembered":
                    peers.append(member.peer())
                continue
            report = member.report
            assert report is not None
            if report.role == "head":
                continue  # its GPUs hold its own model
            if report.owner_id is not None and report.owner_id != self.identity.id:
                continue  # lent to someone else
            if self.config.discovery.require_matching_version and not same_build(
                local_version, report.llamacpp or member.llamacpp
            ):
                log.warning(
                    "membership: not using %s, llama.cpp differs (%s here, %s there)",
                    member.display,
                    local_version,
                    report.llamacpp or member.llamacpp,
                )
                continue
            peers.append(member.peer())
        return peers


ReportProvider = Callable[[], NodeReport]

__all__ = ["Lan", "Member", "Membership", "NodeReport", "ReportProvider"]
