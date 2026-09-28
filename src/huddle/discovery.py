"""Finding cluster peers on the LAN.

Two addresses matter here and they need not be the same one. mDNS is
link-local multicast, so discovery happens over the LAN and the packet's source
address is whatever the interface has. The address a peer should actually be
*dialled* on is carried in the TXT record instead (``connect``): a stable one
such as a Tailscale address when configured, otherwise the node's current LAN
address — which the advertiser re-announces when DHCP moves it.

The TXT record also carries the peer's identity and llama.cpp build:

- ``id`` is a random identity kept in the node's state directory. Hostnames
  collide (cloned images, identical fresh installs), so nodes are told apart
  by this, and a node excludes itself by it.
- ``cluster`` keeps separate groups of machines on one network apart.
- ``llamacpp``: the RPC protocol refuses a version-mismatched peer at connect
  time with an unhelpful error, so catching it here turns a confusing failure
  into a clear one.

Nothing that changes minute to minute goes in TXT — whether a node is serving,
lending or idle. python-zeroconf keeps TXT records for 75 minutes, so a node
that lost power would go on looking busy long after it was gone; that state
is asked for over HTTP instead (see ``membership.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel
from zeroconf import IPVersion, ServiceStateChange
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

from huddle import __version__
from huddle.config import DiscoveryConfig, HuddleConfig, PeerConfig
from huddle.system import default_route_address

log = logging.getLogger("huddle.discovery")

SERVICE_TYPE = "_huddle._tcp.local."
# `huddle serve` reachable only on loopback: it can borrow peers' GPUs but its
# own agent routes are not on the network, so it cannot lend.
ROLE_COORDINATOR = "coordinator"
# `huddle agent`: lends its GPUs, never runs a model itself.
ROLE_WORKER = "worker"
# `huddle serve` on the network: every node equal, either role as needed.
ROLE_NODE = "node"
# Bumped when nodes can no longer understand each other's HTTP API.
PROTO = 1
# How often to check whether DHCP has moved this node's address.
ADDRESS_REFRESH_SECONDS = 30.0


@dataclass(frozen=True)
class Identity:
    """Who this node is, as its peers will know it."""

    id: str
    name: str


class DiscoveredPeer(BaseModel):
    """A node that answered a browse."""

    name: str
    host: str
    agent_port: int
    rpc_port: int
    llamacpp_version: str | None = None
    # Where multicast saw it, which may differ from `host` and is not stable.
    seen_at: str | None = None
    # See the ROLE_* constants. Records from before roles existed carry none.
    role: str | None = None
    # Absent from records written before identities existed.
    id: str | None = None
    cluster: str | None = None
    proto: int | None = None
    huddle_version: str | None = None
    # The mDNS instance name, which is what a goodbye packet refers to.
    service: str | None = None

    @property
    def is_coordinator(self) -> bool:
        return self.role == ROLE_COORDINATOR

    @property
    def key(self) -> str:
        """Identity when the record has one; its name otherwise."""
        return self.id or f"name:{self.name}"

    def to_peer(self) -> PeerConfig:
        return PeerConfig(
            name=self.name,
            host=self.host,
            agent_port=self.agent_port,
            rpc_port=self.rpc_port,
            id=self.id,
        )


def _txt(properties: dict[bytes, bytes | None], key: str) -> str | None:
    raw = properties.get(key.encode())
    return raw.decode(errors="replace") if raw else None


def _safe_label(text: str) -> str:
    # Service names are a DNS label: keep it conservative.
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text)[:40] or "node"


def service_name(identity: Identity) -> str:
    """``<name>-<id6>``: readable, and unique even when two hosts share a name."""
    return f"{_safe_label(identity.name)}-{identity.id[:6]}.{SERVICE_TYPE}"


def advertise_address(config: HuddleConfig) -> str | None:
    """What peers should dial: a configured stable address, else the LAN one."""
    return config.discovery.advertise or config.rpc.advertise or default_route_address()


class Advertiser:
    """Announces this node on the LAN for as long as it is running."""

    def __init__(
        self,
        config: HuddleConfig,
        llamacpp_version: str | None = None,
        *,
        role: str = ROLE_WORKER,
        identity: Identity | None = None,
        port: int | None = None,
        zeroconf: AsyncZeroconf | None = None,
    ) -> None:
        self.config = config
        self.llamacpp_version = llamacpp_version
        self.role = role
        self.identity = identity or _identity(config)
        # Where this node answers /agent/*: the API port for `huddle serve`,
        # which serves them there, and node.agent_port for `huddle agent`.
        self.port = port or config.node.agent_port
        self._shared = zeroconf
        self._zc: AsyncZeroconf | None = None
        self._info: AsyncServiceInfo | None = None
        self._refresh: asyncio.Task[None] | None = None
        self.connect: str | None = None

    def _properties(self, connect: str) -> dict[str, str]:
        properties = {
            "node": self.config.node.name,
            "connect": connect,
            "agent_port": str(self.port),
            "rpc_port": str(self.config.rpc.port),
            "role": self.role,
            "id": self.identity.id,
            "cluster": self.config.discovery.cluster,
            "proto": str(PROTO),
            "huddle": __version__,
        }
        if self.llamacpp_version:
            properties["llamacpp"] = self.llamacpp_version
        return properties

    def _service_info(self, connect: str) -> AsyncServiceInfo:
        address = default_route_address() or "127.0.0.1"
        return AsyncServiceInfo(
            SERVICE_TYPE,
            service_name(self.identity),
            addresses=[socket.inet_aton(address)],
            port=self.port,
            properties=self._properties(connect),
            # Not "<hostname>.local.": avahi already announces that name on
            # Ubuntu, and claiming it too triggers avahi's conflict renaming.
            server=f"huddle-{self.identity.id[:6]}.local.",
        )

    async def start(self) -> None:
        connect = advertise_address(self.config)
        if not connect:
            log.warning(
                "discovery: not advertising: this machine has no LAN address "
                "(set discovery.advertise to the address peers should use)"
            )
            return

        self._zc = self._shared or AsyncZeroconf(ip_version=IPVersion.V4Only)
        self._info = self._service_info(connect)
        await self._zc.async_register_service(self._info, allow_name_change=True)
        self.connect = connect
        log.info(
            "discovery: advertising %s as %s:%d (cluster %r)",
            self.config.node.name,
            connect,
            self.port,
            self.config.discovery.cluster,
        )
        if not (self.config.discovery.advertise or self.config.rpc.advertise):
            self._refresh = asyncio.create_task(self._follow_address())

    async def _follow_address(self) -> None:
        """Re-announce when DHCP gives this node a new address."""
        while True:
            await asyncio.sleep(ADDRESS_REFRESH_SECONDS)
            current = default_route_address()
            if not current or current == self.connect or self._zc is None:
                continue
            log.info("discovery: address moved from %s to %s; re-announcing", self.connect, current)
            info = self._service_info(current)
            try:
                await self._zc.async_update_service(info)
            except Exception as exc:
                log.warning("discovery: could not re-announce: %s", exc)
                continue
            self._info, self.connect = info, current

    async def stop(self) -> None:
        if self._refresh is not None:
            self._refresh.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh
            self._refresh = None
        if self._zc is None:
            return
        with contextlib.suppress(Exception):
            if self._info is not None:
                await self._zc.async_unregister_service(self._info)
            if self._shared is None:
                await self._zc.async_close()
        self._zc, self._info = None, None


def _identity(config: HuddleConfig) -> Identity:
    from huddle import paths
    from huddle.state import StateStore

    node_id = (
        config.node.id or StateStore(config.node.state_dir or paths.state_dir()).load().node_id
    )
    return Identity(id=node_id, name=config.node.name)


def load_identity(config: HuddleConfig) -> Identity:
    """This node's identity: ``node.id`` if set, else the one in its state file."""
    return _identity(config)


# -- browsing ---------------------------------------------------------------------


async def discover(
    config: DiscoveryConfig,
    *,
    exclude: str | None = None,
    exclude_id: str | None = None,
    timeout: float | None = None,
    attempts: int | None = None,
) -> list[DiscoveredPeer]:
    """Browse the LAN for Huddle nodes, retrying until something answers.

    One-shot, for `huddle discover` and the doctor. A running node keeps a
    live view instead (``ZeroconfLan``).

    Retries matter at boot: the service starts shortly after
    network-online.target, but multicast may not work yet and peers may still be
    starting. A single short browse then returns nothing, the cluster plans as a
    single node, and a large model fails to load at all.

    Never raises: discovery failing should leave a cluster running on its static
    peers, not stop it starting.
    """
    tries = attempts if attempts is not None else config.attempts
    for attempt in range(1, tries + 1):
        found = await _browse_once(config, exclude=exclude, exclude_id=exclude_id, timeout=timeout)
        if found or attempt == tries:
            if not found:
                log.info("discovery: nothing found after %d browses", tries)
            return found
        log.info("discovery: nothing found (browse %d/%d), retrying", attempt, tries)
    return []


async def _browse_once(
    config: DiscoveryConfig,
    *,
    exclude: str | None = None,
    exclude_id: str | None = None,
    timeout: float | None = None,
) -> list[DiscoveredPeer]:
    found: dict[str, DiscoveredPeer] = {}
    pending: list[asyncio.Task[None]] = []
    zc = AsyncZeroconf(ip_version=IPVersion.V4Only)

    async def resolve(service_type: str, name: str) -> None:
        peer = await resolve_record(zc, service_type, name)
        if peer is not None and _wanted(peer, config.cluster, exclude, exclude_id):
            found[peer.key] = peer

    def on_change(zeroconf, service_type, name, state_change, **_):  # type: ignore[no-untyped-def]
        if state_change is not ServiceStateChange.Added:
            return
        pending.append(asyncio.create_task(resolve(service_type, name)))

    browser = AsyncServiceBrowser(zc.zeroconf, SERVICE_TYPE, handlers=[on_change])
    try:
        await asyncio.sleep(timeout if timeout is not None else config.timeout)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    finally:
        with contextlib.suppress(Exception):
            await browser.async_cancel()
            await zc.async_close()

    return sorted(found.values(), key=lambda peer: peer.name)


def _wanted(
    peer: DiscoveredPeer, cluster: str, exclude: str | None, exclude_id: str | None
) -> bool:
    if exclude_id is not None and peer.id == exclude_id:
        return False
    # Name-only exclusion is for records without identities, and for callers
    # that do not know their own.
    if exclude is not None and peer.name == exclude and (exclude_id is None or peer.id is None):
        return False
    return (peer.cluster or "default") == cluster


async def resolve_record(zc: AsyncZeroconf, service_type: str, name: str) -> DiscoveredPeer | None:
    info = AsyncServiceInfo(service_type, name)
    if not await info.async_request(zc.zeroconf, 3000):
        return None

    properties: dict[bytes, bytes | None] = info.properties or {}
    node = _txt(properties, "node")
    connect = _txt(properties, "connect")
    if not node or not connect:
        return None

    addresses = info.parsed_addresses()
    proto = _txt(properties, "proto")
    return DiscoveredPeer(
        name=node,
        host=connect,
        agent_port=int(_txt(properties, "agent_port") or info.port or 8081),
        rpc_port=int(_txt(properties, "rpc_port") or 50052),
        llamacpp_version=_txt(properties, "llamacpp"),
        seen_at=addresses[0] if addresses else None,
        role=_txt(properties, "role"),
        id=_txt(properties, "id"),
        cluster=_txt(properties, "cluster"),
        proto=int(proto) if proto and proto.isdigit() else None,
        huddle_version=_txt(properties, "huddle"),
        service=name,
    )


# -- a live view, for a running node ------------------------------------------------

OnUp = Callable[[DiscoveredPeer], None]
OnDown = Callable[[str], None]


class ZeroconfLan:
    """This node's advertisement plus a browser that never stops.

    One ``AsyncZeroconf`` for both. Every record seen or updated is reported
    through ``on_up`` (filtering is the membership table's job); a goodbye is
    reported through ``on_down`` with the service name it refers to.
    """

    def __init__(self, advertiser: Advertiser | None = None) -> None:
        self.advertiser = advertiser
        self._zc: AsyncZeroconf | None = None
        self._browser: AsyncServiceBrowser | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self, on_up: OnUp, on_down: OnDown) -> None:
        self._zc = AsyncZeroconf(ip_version=IPVersion.V4Only)
        zc = self._zc
        if self.advertiser is not None:
            self.advertiser._shared = zc
            try:
                await self.advertiser.start()
            except Exception as exc:
                # A node that cannot advertise can still find and use others.
                log.error("discovery: could not advertise this node: %s", exc)

        async def report(service_type: str, name: str) -> None:
            peer = await resolve_record(zc, service_type, name)
            if peer is not None:
                on_up(peer)

        def on_change(zeroconf, service_type, name, state_change, **_):  # type: ignore[no-untyped-def]
            if state_change is ServiceStateChange.Removed:
                on_down(name)
                return
            task = asyncio.create_task(report(service_type, name))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        self._browser = AsyncServiceBrowser(zc.zeroconf, SERVICE_TYPE, handlers=[on_change])

    async def stop(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        with contextlib.suppress(Exception):
            if self._browser is not None:
                await self._browser.async_cancel()
        if self.advertiser is not None:
            await self.advertiser.stop()
        with contextlib.suppress(Exception):
            if self._zc is not None:
                await self._zc.async_close()
        self._zc, self._browser = None, None


__all__ = [
    "PROTO",
    "ROLE_COORDINATOR",
    "ROLE_NODE",
    "ROLE_WORKER",
    "SERVICE_TYPE",
    "Advertiser",
    "DiscoveredPeer",
    "Identity",
    "ZeroconfLan",
    "advertise_address",
    "discover",
    "load_identity",
    "resolve_record",
    "service_name",
]
