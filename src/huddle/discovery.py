"""Finding cluster peers on the LAN.

Two addresses matter here and they are not the same one. mDNS is link-local
multicast, so discovery happens over the LAN and the packet's source address is
a DHCP lease that moves — we watched maksood shift from .25 to .38 mid-session.
The address a peer should actually be *dialled* on is carried in the TXT record
instead, which is how discovery and stable addressing coexist: find nodes by
multicast, talk to them over Tailscale.

The TXT record also carries the peer's llama.cpp version. The RPC protocol
refuses a version-mismatched peer at connect time with an unhelpful error, so
catching it during discovery turns a confusing failure into a clear one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket

from pydantic import BaseModel
from zeroconf import IPVersion, ServiceStateChange
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

from huddle.config import DiscoveryConfig, HuddleConfig, PeerConfig

log = logging.getLogger("huddle.discovery")

SERVICE_TYPE = "_huddle._tcp.local."


class DiscoveredPeer(BaseModel):
    """A node that answered a browse."""

    name: str
    host: str
    agent_port: int
    rpc_port: int
    llamacpp_version: str | None = None
    # Where multicast saw it, which may differ from `host` and is not stable.
    seen_at: str | None = None

    def to_peer(self) -> PeerConfig:
        return PeerConfig(
            name=self.name,
            host=self.host,
            agent_port=self.agent_port,
            rpc_port=self.rpc_port,
        )


def _txt(properties: dict[bytes, bytes | None], key: str) -> str | None:
    raw = properties.get(key.encode())
    return raw.decode(errors="replace") if raw else None


def _service_name(node_name: str) -> str:
    # Service names are a DNS label: keep it conservative.
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in node_name)
    return f"{safe}.{SERVICE_TYPE}"


class Advertiser:
    """Announces this node on the LAN for as long as it is running."""

    def __init__(self, config: HuddleConfig, llamacpp_version: str | None = None) -> None:
        self.config = config
        self.llamacpp_version = llamacpp_version
        self._zc: AsyncZeroconf | None = None
        self._info: AsyncServiceInfo | None = None

    async def start(self) -> None:
        connect_host = self.config.discovery.advertise or self.config.rpc.advertise
        if not connect_host:
            # Without a stable address the only thing we could advertise is a
            # DHCP lease, which is the problem discovery is meant to avoid.
            log.warning(
                "discovery: not advertising, no stable address configured "
                "(set discovery.advertise or rpc.advertise)"
            )
            return

        properties: dict[str, str] = {
            "node": self.config.node.name,
            "connect": connect_host,
            "agent_port": str(self.config.node.agent_port),
            "rpc_port": str(self.config.rpc.port),
        }
        if self.llamacpp_version:
            properties["llamacpp"] = self.llamacpp_version

        self._zc = AsyncZeroconf(ip_version=IPVersion.V4Only)
        self._info = AsyncServiceInfo(
            SERVICE_TYPE,
            _service_name(self.config.node.name),
            addresses=[socket.inet_aton(_local_address())],
            port=self.config.node.agent_port,
            properties=properties,
            server=f"{self.config.node.name}.local.",
        )
        await self._zc.async_register_service(self._info)
        log.info("discovery: advertising %s as %s", self.config.node.name, connect_host)

    async def stop(self) -> None:
        if self._zc is None:
            return
        with contextlib.suppress(Exception):
            if self._info is not None:
                await self._zc.async_unregister_service(self._info)
            await self._zc.async_close()
        self._zc, self._info = None, None


def _local_address() -> str:
    """This host's LAN address, for the mDNS A record.

    Only used so the record is well-formed; peers connect using the `connect`
    TXT value instead.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, contextlib.suppress(OSError):
        sock.connect(("192.0.2.1", 1))  # TEST-NET-1: routes nowhere, sends nothing
        return str(sock.getsockname()[0])
    return "127.0.0.1"


async def discover(
    config: DiscoveryConfig,
    *,
    exclude: str | None = None,
    timeout: float | None = None,
    attempts: int | None = None,
) -> list[DiscoveredPeer]:
    """Browse the LAN for Huddle nodes, retrying until something answers.

    Retries matter at boot: the service starts shortly after
    network-online.target, but multicast may not work yet and peers may still be
    starting. A single short browse then returns nothing, the cluster plans as a
    single node, and a large model fails to load at all.

    Never raises: discovery failing should leave a cluster running on its static
    peers, not stop it starting.
    """
    tries = attempts if attempts is not None else config.attempts
    for attempt in range(1, tries + 1):
        found = await _browse_once(config, exclude=exclude, timeout=timeout)
        if found or attempt == tries:
            if not found:
                log.info("discovery: nothing found after %d browses", tries)
            return found
        log.info("discovery: nothing found (browse %d/%d), retrying", attempt, tries)
    return []


async def _browse_once(
    config: DiscoveryConfig, *, exclude: str | None = None, timeout: float | None = None
) -> list[DiscoveredPeer]:
    found: dict[str, DiscoveredPeer] = {}
    pending: list[asyncio.Task[None]] = []
    zc = AsyncZeroconf(ip_version=IPVersion.V4Only)

    def on_change(zeroconf, service_type, name, state_change, **_):  # type: ignore[no-untyped-def]
        if state_change is not ServiceStateChange.Added:
            return
        pending.append(asyncio.create_task(_resolve(zc, service_type, name, found, exclude)))

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


async def _resolve(
    zc: AsyncZeroconf,
    service_type: str,
    name: str,
    found: dict[str, DiscoveredPeer],
    exclude: str | None,
) -> None:
    info = AsyncServiceInfo(service_type, name)
    if not await info.async_request(zc.zeroconf, 3000):
        return

    properties: dict[bytes, bytes | None] = info.properties or {}
    node = _txt(properties, "node")
    connect = _txt(properties, "connect")
    if not node or not connect or node == exclude:
        return

    addresses = info.parsed_addresses()
    found[node] = DiscoveredPeer(
        name=node,
        host=connect,
        agent_port=int(_txt(properties, "agent_port") or info.port or 8081),
        rpc_port=int(_txt(properties, "rpc_port") or 50052),
        llamacpp_version=_txt(properties, "llamacpp"),
        seen_at=addresses[0] if addresses else None,
    )
