"""Peer discovery.

The property that matters is the two-address split: nodes are *found* over
link-local multicast but *dialled* on the stable address they advertise. Getting
that backwards reintroduces exactly the DHCP drift discovery is meant to remove.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from huddle.config import DiscoveryConfig, HuddleConfig
from huddle.coordinator.cluster import resolve_peers
from huddle.discovery import DiscoveredPeer

SAME_COMMIT = "version: 0.4.1-dev (build 10975, commit 4c9233c03)"
# Same commit, lower build number: what a --depth=1 clone reports.
SHALLOW_SAME_COMMIT = "version: 0.4.1-dev (build 1, commit 4c9233c)"
OTHER_COMMIT = "version: 0.4.1-dev (build 10975, commit deadbeef)"


def peer(name: str, host: str, **kw: object) -> DiscoveredPeer:
    return DiscoveredPeer(name=name, host=host, agent_port=8081, rpc_port=50052, **kw)  # type: ignore[arg-type]


def fake_discover(result: list[DiscoveredPeer]) -> Callable[..., object]:
    async def _discover(*_args: object, **_kw: object) -> list[DiscoveredPeer]:
        return result

    return _discover


async def test_disabled_discovery_uses_only_configured_peers(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "huddle.coordinator.cluster.discover", fake_discover([peer("ghost", "10.0.0.9")])
    )
    huddle_config.discovery.enabled = False
    assert await resolve_peers(huddle_config) == []


async def test_discovered_peers_are_added(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "huddle.coordinator.cluster.discover",
        fake_discover([peer("box2", "100.64.0.2"), peer("box3", "100.64.0.3")]),
    )
    huddle_config.discovery.enabled = True
    resolved = await resolve_peers(huddle_config)

    assert [p.name for p in resolved] == ["box2", "box3"]
    # Dialled on the advertised address, not whatever multicast saw.
    assert resolved[0].rpc_endpoint == "100.64.0.2:50052"


async def test_configured_peer_wins_over_discovery(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writing an address down is a decision; discovery must not override it."""
    huddle_config.peers = [
        HuddleConfig.model_validate(
            {
                "binaries": {"llama_server": str(huddle_config.binaries.llama_server)},
                "models": {"dir": str(huddle_config.models.dir)},
                "peers": [{"name": "box2", "host": "192.168.1.50", "rpc_port": 51000}],
            }
        ).peers[0]
    ]
    monkeypatch.setattr(
        "huddle.coordinator.cluster.discover", fake_discover([peer("box2", "100.64.0.2")])
    )
    huddle_config.discovery.enabled = True
    resolved = await resolve_peers(huddle_config)

    assert len(resolved) == 1
    assert resolved[0].host == "192.168.1.50"
    assert resolved[0].rpc_port == 51000


async def test_version_mismatched_peer_is_skipped(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The RPC handshake rejects these anyway, but later and less clearly."""
    monkeypatch.setattr(
        "huddle.coordinator.cluster.discover",
        fake_discover(
            [
                peer("same", "100.64.0.2", llamacpp_version="build 10975"),
                peer("other", "100.64.0.3", llamacpp_version=OTHER_COMMIT),
            ]
        ),
    )
    huddle_config.discovery.enabled = True
    resolved = await resolve_peers(huddle_config, SAME_COMMIT)
    assert [p.name for p in resolved] == ["same"]


async def test_version_check_can_be_disabled(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "huddle.coordinator.cluster.discover",
        fake_discover([peer("other", "100.64.0.3", llamacpp_version=OTHER_COMMIT)]),
    )
    huddle_config.discovery.enabled = True
    huddle_config.discovery.require_matching_version = False
    assert [p.name for p in await resolve_peers(huddle_config, SAME_COMMIT)] == ["other"]


async def test_discovery_failure_falls_back_to_configured_peers(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken network must not stop a cluster starting on its static peers."""

    async def boom(*_args: object, **_kw: object) -> list[DiscoveredPeer]:
        raise OSError("no multicast route")

    monkeypatch.setattr("huddle.coordinator.cluster.discover", boom)
    huddle_config.discovery.enabled = True
    assert await resolve_peers(huddle_config) == []


async def test_advertise_and_discover_round_trip(huddle_config: HuddleConfig) -> None:
    """A real mDNS round trip, to prove the TXT contract end to end."""
    from huddle.discovery import Advertiser, discover

    huddle_config.node.name = "roundtrip-node"
    huddle_config.discovery.enabled = True
    huddle_config.discovery.advertise = "100.64.99.1"
    huddle_config.rpc.port = 50099

    advertiser = Advertiser(huddle_config, llamacpp_version="build test")
    await advertiser.start()
    try:
        found = await discover(DiscoveryConfig(timeout=4.0), exclude="someone-else")
    finally:
        await advertiser.stop()

    ours = [p for p in found if p.name == "roundtrip-node"]
    if not ours:
        pytest.skip("no multicast on this host; logic is covered by the tests above")

    got = ours[0]
    assert got.host == "100.64.99.1", "must dial the advertised address"
    assert got.rpc_port == 50099
    assert got.llamacpp_version == "build test"


async def test_self_is_excluded_from_discovery(huddle_config: HuddleConfig) -> None:
    from huddle.discovery import Advertiser, discover

    huddle_config.node.name = "myself"
    huddle_config.discovery.advertise = "100.64.99.2"
    advertiser = Advertiser(huddle_config, llamacpp_version="v")
    await advertiser.start()
    try:
        found = await discover(DiscoveryConfig(timeout=3.0), exclude="myself")
    finally:
        await advertiser.stop()

    assert not any(p.name == "myself" for p in found)


async def test_shallow_clone_build_number_is_not_a_mismatch(
    huddle_config: HuddleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same commit, different build number, because one node used --depth=1.

    This broke a working cluster: the build number counts commits since the root,
    so clone depth changes it even when the source is identical. Compatibility
    depends on the commit, which is what the RPC handshake actually cares about.
    """
    monkeypatch.setattr(
        "huddle.coordinator.cluster.discover",
        fake_discover([peer("maksood", "100.98.227.49", llamacpp_version=SAME_COMMIT)]),
    )
    huddle_config.discovery.enabled = True
    resolved = await resolve_peers(huddle_config, SHALLOW_SAME_COMMIT)
    assert [p.name for p in resolved] == ["maksood"], "same commit must be compatible"


async def test_discovery_retries_before_giving_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """At boot the first browse often finds nothing; a single try is not enough."""
    from huddle import discovery

    calls = {"n": 0}

    async def flaky(*_args: object, **_kw: object) -> list[DiscoveredPeer]:
        calls["n"] += 1
        return [peer("late", "100.64.0.5")] if calls["n"] >= 3 else []

    monkeypatch.setattr(discovery, "_browse_once", flaky)
    found = await discovery.discover(DiscoveryConfig(attempts=4, timeout=0.01))

    assert [p.name for p in found] == ["late"]
    assert calls["n"] == 3, "should stop as soon as something answers"


async def test_discovery_gives_up_after_configured_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huddle import discovery

    calls = {"n": 0}

    async def never(*_args: object, **_kw: object) -> list[DiscoveredPeer]:
        calls["n"] += 1
        return []

    monkeypatch.setattr(discovery, "_browse_once", never)
    assert await discovery.discover(DiscoveryConfig(attempts=3, timeout=0.01)) == []
    assert calls["n"] == 3
