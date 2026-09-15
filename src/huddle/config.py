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
