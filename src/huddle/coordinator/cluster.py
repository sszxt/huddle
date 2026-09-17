"""Gathering what the cluster has to offer.

The coordinator asks each peer's agent what it can contribute, then arranges
every device into the order llama.cpp will enumerate them. That ordering is the
whole point: ``--tensor-split`` is positional, and a peer's devices appear in
the order its endpoint was passed to ``--rpc``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

from huddle.config import HuddleConfig, PeerConfig
from huddle.coordinator.planner import PlacementDevice
from huddle.discovery import DiscoveredPeer, discover
from huddle.hardware import NodeHardware
from huddle.llamacpp import same_build

log = logging.getLogger("huddle.cluster")


@dataclass
class PeerReport:
    """What one peer said when asked, or why it could not answer."""

    peer: PeerConfig
    hardware: NodeHardware | None = None
    error: str | None = None

    @property
    def reachable(self) -> bool:
        return self.hardware is not None


async def query_peer(
    client: httpx.AsyncClient, peer: PeerConfig, *, timeout: float = 10.0
) -> PeerReport:
    """Ask one peer's agent to describe itself.

    Never raises: an unreachable peer is reported, not fatal. A cluster that
    refuses to start because one node is down is worse than one that starts
    smaller and says so.
    """
    url = f"http://{peer.host}:{peer.agent_port}/agent/hardware"
    try:
        response = await client.get(url, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return PeerReport(peer=peer, error=f"{type(exc).__name__}: {exc}")
    return PeerReport(peer=peer, hardware=NodeHardware.model_validate(response.json()))


async def resolve_peers(config: HuddleConfig, local_version: str | None = None) -> list[PeerConfig]:
    """The peers to use: those configured, plus any found on the LAN.

    A configured peer always wins over a discovered one of the same name —
    writing an address down is a decision, and discovery should not quietly
    override it.
    """
    peers = list(config.peers)
    if not config.discovery.enabled:
        return peers

    known = {peer.name for peer in peers}
    try:
        discovered = await discover(config.discovery, exclude=config.node.name)
    except Exception as exc:
        log.warning("discovery failed, using configured peers only: %s", exc)
        return peers

    for candidate in discovered:
        if candidate.is_coordinator:
            # Another head node. Its agent routes are not on the advertised
            # port and its GPUs belong to its own model.
            log.debug("discovery: %s is a coordinator, not a worker; skipping", candidate.name)
            continue
        if candidate.name in known:
            log.info(
                "discovery: %s already configured, keeping the configured entry", candidate.name
            )
            continue
        if _version_mismatch(config, local_version, candidate):
            continue
        log.info("discovery: found %s at %s", candidate.name, candidate.host)
        peers.append(candidate.to_peer())
    return peers


def _version_mismatch(
    config: HuddleConfig, local_version: str | None, candidate: DiscoveredPeer
) -> bool:
    """Whether to skip a peer built from a different llama.cpp.

    The RPC handshake would reject it anyway, but only at connect time and with
    an error that does not name the real problem.
    """
    if not config.discovery.require_matching_version:
        return False
    if same_build(local_version, candidate.llamacpp_version):
        return False
    log.warning(
        "discovery: skipping %s, llama.cpp version differs (%s here, %s there)",
        candidate.name,
        local_version,
        candidate.llamacpp_version,
    )
    return True


async def query_peers(config: HuddleConfig, *, timeout: float = 10.0) -> list[PeerReport]:
    """Ask every peer in parallel, preserving order."""
    peers = await resolve_peers(config, _local_llamacpp_version(config))
    if not peers:
        return []
    async with httpx.AsyncClient() as client:
        return list(
            await asyncio.gather(*(query_peer(client, peer, timeout=timeout) for peer in peers))
        )


def _local_llamacpp_version(config: HuddleConfig) -> str | None:
    from huddle.llamacpp import LlamaCppError, binary_version

    try:
        return binary_version(config.binaries.llama_server)
    except LlamaCppError:
        return None


def build_device_list(
    local: NodeHardware, reports: list[PeerReport]
) -> tuple[list[PlacementDevice], list[str]]:
    """Return devices in llama.cpp enumeration order, plus the ``--rpc`` endpoints.

    Remote devices come first, grouped by peer in the order those endpoints are
    passed to ``--rpc``; local devices follow. One peer contributes *one*
    endpoint but as many RPC devices as it has GPUs.
    """
    remote: list[PlacementDevice] = []
    endpoints: list[str] = []

    for report in reports:
        if report.hardware is None:
            continue
        endpoints.append(report.peer.rpc_endpoint)
        for device in report.hardware.devices:
            if device.is_rpc:
                continue  # a peer must not re-export someone else's devices
            remote.append(
                PlacementDevice(
                    id=f"RPC{len(remote)}",
                    name=device.name,
                    node=report.peer.name,
                    free_mib=device.free_mib or device.total_mib or 0,
                    unified_memory=device.unified_memory,
                    is_rpc=True,
                )
            )

    local_devices = [
        PlacementDevice(
            id=device.id,
            name=device.name,
            node=local.name,
            free_mib=device.free_mib or device.total_mib or 0,
            unified_memory=device.unified_memory,
            is_rpc=False,
        )
        for device in local.devices
        if not device.is_rpc
    ]

    return remote + local_devices, endpoints
