"""A LAN in memory, standing in for mDNS in tests.

Real multicast is unreliable on CI runners and dev boxes (test_discovery skips
when it is missing), and this is about membership logic, not zeroconf. Every
node started on one hub sees every other node's advertisement, including the
ones that were there before it, and a goodbye when one stops — the same
events ``ZeroconfLan`` produces.
"""

from __future__ import annotations

from huddle import __version__
from huddle.discovery import (
    PROTO,
    Advertiser,
    DiscoveredPeer,
    OnDown,
    OnUp,
    advertise_address,
    service_name,
)


def advert_of(advertiser: Advertiser) -> DiscoveredPeer:
    """What the rest of the LAN would read from this node's TXT record."""
    config = advertiser.config
    return DiscoveredPeer(
        name=config.node.name,
        host=advertise_address(config) or "127.0.0.1",
        agent_port=advertiser.port,
        rpc_port=config.rpc.port,
        llamacpp_version=advertiser.llamacpp_version,
        role=advertiser.role,
        id=advertiser.identity.id,
        cluster=config.discovery.cluster,
        proto=PROTO,
        huddle_version=__version__,
        service=service_name(advertiser.identity),
    )


class LanHub:
    """One network. Nodes join by starting a ``MemoryLan`` on it."""

    def __init__(self) -> None:
        self.nodes: list[MemoryLan] = []
        # Advertisements with no node behind them in this process, e.g. a
        # `huddle agent` served by a fixture.
        self.static: list[DiscoveredPeer] = []

    def lan(self, advertiser: Advertiser) -> MemoryLan:
        return MemoryLan(self, advertiser)

    def announce(self, peer: DiscoveredPeer) -> None:
        self.static.append(peer)
        for node in self.nodes:
            node.deliver(peer)

    def withdraw(self, peer: DiscoveredPeer) -> None:
        self.static.remove(peer)
        for node in self.nodes:
            if node.on_down is not None and peer.service:
                node.on_down(peer.service)


class MemoryLan:
    def __init__(self, hub: LanHub, advertiser: Advertiser | None = None) -> None:
        self.hub = hub
        self.advertiser = advertiser
        self.advert: DiscoveredPeer | None = None
        self.on_up: OnUp | None = None
        self.on_down: OnDown | None = None

    def deliver(self, peer: DiscoveredPeer) -> None:
        if self.on_up is not None:
            self.on_up(peer)

    async def start(self, on_up: OnUp, on_down: OnDown) -> None:
        self.on_up, self.on_down = on_up, on_down
        if self.advertiser is not None:
            self.advert = advert_of(self.advertiser)
        for peer in self.hub.static:
            self.deliver(peer)
        for node in self.hub.nodes:
            if node.advert is not None:
                self.deliver(node.advert)
            if self.advert is not None:
                node.deliver(self.advert)
        self.hub.nodes.append(self)

    async def stop(self) -> None:
        if self in self.hub.nodes:
            self.hub.nodes.remove(self)
        if self.advert is not None and self.advert.service:
            for node in self.hub.nodes:
                if node.on_down is not None:
                    node.on_down(self.advert.service)
        self.on_up = self.on_down = None
