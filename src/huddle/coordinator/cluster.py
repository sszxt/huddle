"""Gathering what the cluster has to offer.

The coordinator asks each peer's agent what it can contribute, then arranges
every device into the order llama.cpp will enumerate them. That ordering is the
whole point: ``--tensor-split`` is positional, and a peer's devices appear in
the order its endpoint was passed to ``--rpc``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from huddle.config import HuddleConfig, PeerConfig
from huddle.coordinator.planner import PlacementDevice
from huddle.hardware import NodeHardware


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


async def query_peers(config: HuddleConfig, *, timeout: float = 10.0) -> list[PeerReport]:
    """Ask every peer in parallel, preserving configured order."""
    if not config.peers:
        return []
    async with httpx.AsyncClient() as client:
        return list(
            await asyncio.gather(
                *(query_peer(client, peer, timeout=timeout) for peer in config.peers)
            )
        )


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
