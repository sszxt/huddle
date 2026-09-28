"""Where an installed Huddle keeps things, and how it finds llama.cpp.

The XDG base directories, so nothing depends on the working directory: config
in ``~/.config/huddle``, binaries and models in ``~/.local/share/huddle``, and
the node's identity in ``~/.local/state/huddle``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huddle.pin import load_pin

# Upstream builds the worker as ggml-rpc-server; its README still says rpc-server.
RPC_SERVER_NAMES = ("ggml-rpc-server", "rpc-server")


def _xdg(variable: str, fallback: str) -> Path:
    value = os.environ.get(variable)
    base = Path(value) if value else Path.home() / fallback
    return base / "huddle"


def config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config")


def data_dir() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share")


def state_dir() -> Path:
    return _xdg("XDG_STATE_HOME", ".local/state")


def default_config_file() -> Path:
    return config_dir() / "huddle.yaml"


def default_models_dir() -> Path:
    return data_dir() / "models"


def llamacpp_root() -> Path:
    """Parent of every installed llama.cpp release, one directory each."""
    return data_dir() / "llama.cpp"


def installed_llamacpp() -> Path | None:
    """The directory `huddle setup` installs the pinned release into."""
    pin = load_pin()
    if pin is None or pin.release is None:
        return None
    return llamacpp_root() / pin.release


def find_llama_server() -> Path:
    """The pinned release if installed, else whatever is on PATH.

    Falls back to the install location even when nothing is there, so the
    error a user eventually sees names the path `huddle setup` would fill.
    """
    installed = installed_llamacpp()
    if installed is not None and (installed / "llama-server").is_file():
        return installed / "llama-server"
    on_path = shutil.which("llama-server")
    if on_path:
        return Path(on_path)
    return (installed or llamacpp_root()) / "llama-server"


def find_rpc_server(llama_server: Path) -> Path | None:
    """The worker binary beside ``llama_server``.

    Only beside it: a worker from any other build would be refused by the RPC
    handshake, so a match found elsewhere on PATH would be worse than none.
    """
    for name in RPC_SERVER_NAMES:
        candidate = llama_server.parent / name
        if candidate.is_file():
            return candidate
    return None


__all__ = [
    "RPC_SERVER_NAMES",
    "config_dir",
    "data_dir",
    "default_config_file",
    "default_models_dir",
    "find_llama_server",
    "find_rpc_server",
    "installed_llamacpp",
    "llamacpp_root",
    "state_dir",
]
