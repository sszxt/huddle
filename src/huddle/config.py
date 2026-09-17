"""Node and cluster configuration.

Binary paths and model directories are always configured, never hardcoded: they
differ between the dev box (where llama.cpp is not installed at all) and each
target box. Config loads from YAML, with ``HUDDLE_*`` environment overrides for
the handful of values that change per invocation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_RPC_PORT = 50052
DEFAULT_AGENT_PORT = 8081
DEFAULT_BACKEND_PORT = 8080
DEFAULT_API_PORT = 8000


class Model(BaseModel):
    """Base for config sections: reject unknown keys so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class BinariesConfig(Model):
    """Paths to this node's llama.cpp binaries.

    Both must come from the *same* llama.cpp build. The RPC protocol performs a
    version handshake and refuses mismatched peers, so a cluster running mixed
    builds fails at connect time with a version-mismatch error.
    """

    llama_server: Path
    rpc_server: Path | None = None


class NodeConfig(Model):
    """Identity and agent bind address for this machine."""

    name: str = Field(default_factory=lambda: os.uname().nodename)
    agent_host: str = "127.0.0.1"
    agent_port: int = DEFAULT_AGENT_PORT


class ModelsConfig(Model):
    """Where GGUF files live on this node.

    Only the head node needs the model: llama.cpp reads it locally and streams
    tensors to RPC peers, so there is nothing to distribute to workers.
    """

    dir: Path
    default: str | None = None

    def resolve(self, name: str | None = None) -> Path:
        """Return the absolute path of a model by file name."""
        chosen = name or self.default
        if chosen is None:
            raise ValueError("no model requested and no default configured")
        candidate = Path(chosen)
        return candidate if candidate.is_absolute() else self.dir / candidate


class BackendConfig(Model):
    """How the head ``llama-server`` process is launched."""

    host: str = "127.0.0.1"
    port: int = DEFAULT_BACKEND_PORT
    ctx_size: int = 4096
    parallel: int = 1
    # Accepts an exact layer count, "auto", or "all" (llama.cpp upstream syntax).
    n_gpu_layers: int | str = "all"
    api_key: str | None = None
    autostart: bool = True
    # Full llama.cpp output on disk (llama-server --log-file). Verbose output
    # overwhelms journald and our in-memory buffers, and a failed load is exactly
    # when the whole log is needed. Off until confirmed on the pinned build.
    log_file: Path | None = None
    extra_args: list[str] = Field(default_factory=list)


class RpcConfig(Model):
    """How this node's ``rpc-server`` is launched.

    ``bind`` defaults to ``0.0.0.0`` rather than llama.cpp's own ``127.0.0.1``.
    Upstream's default makes a worker unreachable from other machines in a way
    that looks exactly like a firewall problem, so Huddle is explicit.

    ``cache`` is on by default: without it every model load re-streams the full
    weights over the network, which on 1 GbE costs minutes.
    """

    bind: str = "0.0.0.0"
    port: int = DEFAULT_RPC_PORT
    cache: bool = True
    devices: list[str] = Field(default_factory=list)
    threads: int | None = None
    # What peers should dial to reach this node. Bind is 0.0.0.0, which is not
    # an address anyone can connect to, so report something routable instead —
    # on this cluster, the node's Tailscale address.
    advertise: str | None = None


class PlannerConfig(Model):
    """How conservatively to fill devices.

    ``headroom`` is the fraction of a device's free memory left for compute
    buffers and fragmentation. Overshooting is not a slow cluster, it is an
    out-of-memory failure at load, so the default is deliberately generous.
    """

    headroom: float = Field(default=0.15, ge=0.0, lt=1.0)
    # Fixed margin per device, on top of `headroom`. Zero by default because the
    # proportional headroom alone is what the real two-node cluster was verified
    # with; raise it if loads fail near the limit.
    reserve_mib: float = Field(default=0.0, ge=0.0)
    # Integrated GPUs advertise system RAM as if it were VRAM. Enabling this
    # takes them at their word, which is rarely what you want.
    use_unified_memory: bool = False
    # A plan is an estimate. When a load runs out of device memory anyway, replan
    # with more margin on the device that failed, this many times.
    oom_retries: int = Field(default=2, ge=0)
    # How much margin each retry adds. 1024 MiB mirrors llama.cpp's own
    # --fit-target default, upstream's figure for a per-device safety margin.
    oom_step_mib: float = Field(default=1024.0, gt=0.0)


class DiscoveryConfig(Model):
    """Finding peers on the LAN instead of listing them.

    Discovered peers are merged with any configured under ``peers``; an explicit
    entry always wins, because someone who wrote an address down meant it.
    """

    enabled: bool = False
    timeout: float = Field(default=3.0, gt=0)
    # Browses to make before giving up. At boot the network may not be ready and
    # peers may still be booting, so a single short browse finds nothing and the
    # cluster silently plans as a single node — which for a large model means an
    # out-of-memory failure rather than a small cluster.
    attempts: int = Field(default=3, ge=1)
    # The address peers should dial us on, which is deliberately not the address
    # multicast sees: that one is a DHCP lease and moves. Defaults to
    # ``rpc.advertise`` when unset.
    advertise: str | None = None
    # Refuse peers built from a different llama.cpp. The RPC handshake rejects
    # them anyway, but at connect time with an unhelpful message.
    require_matching_version: bool = True


class SupervisorConfig(Model):
    """How hard to try to keep a wanted cluster running."""

    watch_interval: float = Field(default=5.0, gt=0)
    # Restarts allowed before giving up. With backoff, 10 attempts span roughly
    # twenty minutes — long enough to ride out a peer rebooting.
    max_restarts: int = Field(default=10, ge=0)
    # Failed restarts back off exponentially up to this many seconds, so a fast
    # failing cause does not burn every attempt in half a minute.
    backoff_max: float = Field(default=300.0, gt=0)
    # Running this long without dying forgives earlier restarts. Without it a
    # long-lived cluster with occasional crashes eventually exhausts the limit
    # and stops recovering for good.
    stable_after: float = Field(default=600.0, gt=0)
    # While the cluster is coming up (after a start or a crash), a plan that
    # leaves out an unreachable known peer is not accepted for this long.
    # Otherwise a peer that is merely slower to boot shrinks the cluster — seen
    # on the real nodes: a 32B came up with 34 of 64 layers on CPU.
    peer_wait: float = Field(default=120.0, ge=0)
    # While layers run on CPU, look this often for reachable peers the plan is
    # not using, and replan to include them. 0 disables.
    rejoin_interval: float = Field(default=60.0, ge=0)


class PeerConfig(Model):
    """A remote node participating in the cluster."""

    name: str
    host: str
    agent_port: int = DEFAULT_AGENT_PORT
    rpc_port: int = DEFAULT_RPC_PORT

    @property
    def rpc_endpoint(self) -> str:
        """``host:port`` as llama.cpp's ``--rpc`` flag expects it."""
        return f"{self.host}:{self.rpc_port}"


class ApiConfig(Model):
    """The public OpenAI-compatible API."""

    host: str = "127.0.0.1"
    port: int = DEFAULT_API_PORT
    api_key: str | None = None


class HuddleConfig(Model):
    """Top-level configuration for one node."""

    node: NodeConfig = Field(default_factory=NodeConfig)
    binaries: BinariesConfig
    models: ModelsConfig
    backend: BackendConfig = Field(default_factory=BackendConfig)
    rpc: RpcConfig = Field(default_factory=RpcConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)
    peers: list[PeerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_peer_names_unique(self) -> HuddleConfig:
        names = [peer.name for peer in self.peers]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate peer names: {', '.join(sorted(duplicates))}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> HuddleConfig:
        """Load config from a YAML file, applying environment overrides."""
        path = Path(path)
        with path.open() as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls.model_validate(_apply_env_overrides(raw))


# Environment overrides, kept deliberately small: only values that legitimately
# vary per invocation rather than per machine.
_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "HUDDLE_MODEL": ("models", "default"),
    "HUDDLE_API_PORT": ("api", "port"),
    "HUDDLE_AGENT_PORT": ("node", "agent_port"),
    "HUDDLE_RPC_PORT": ("rpc", "port"),
}


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``HUDDLE_*`` environment variables onto a parsed config dict."""
    merged = {key: dict(value) if isinstance(value, dict) else value for key, value in raw.items()}
    for env_name, (section, key) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value is None:
            continue
        merged.setdefault(section, {})[key] = value
    return merged
