from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from huddle.config import HuddleConfig

MINIMAL: dict[str, Any] = {
    "binaries": {"llama_server": "/opt/llama.cpp/llama-server"},
    "models": {"dir": "/models", "default": "qwen.gguf"},
}


def write_config(tmp_path: Path, data: Mapping[str, Any]) -> Path:
    path = tmp_path / "huddle.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_loads_minimal_config(tmp_path: Path) -> None:
    config = HuddleConfig.load(write_config(tmp_path, MINIMAL))
    assert config.binaries.llama_server == Path("/opt/llama.cpp/llama-server")
    assert config.models.default == "qwen.gguf"


def test_rpc_bind_defaults_to_all_interfaces(tmp_path: Path) -> None:
    """Huddle deliberately departs from llama.cpp's 127.0.0.1 default."""
    config = HuddleConfig.load(write_config(tmp_path, MINIMAL))
    assert config.rpc.bind == "0.0.0.0"
    assert config.rpc.cache is True


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    """A typo in YAML should fail loudly, not be silently ignored."""
    data = {**MINIMAL, "backend": {"prot": 9999}}
    with pytest.raises(ValidationError):
        HuddleConfig.load(write_config(tmp_path, data))


def test_duplicate_peer_names_rejected(tmp_path: Path) -> None:
    data = {
        **MINIMAL,
        "peers": [
            {"name": "box", "host": "192.168.0.54"},
            {"name": "box", "host": "192.168.0.55"},
        ],
    }
    with pytest.raises(ValidationError, match="duplicate peer names"):
        HuddleConfig.load(write_config(tmp_path, data))


def test_peer_rpc_endpoint_format(tmp_path: Path) -> None:
    data = {**MINIMAL, "peers": [{"name": "box", "host": "192.168.0.54", "rpc_port": 50100}]}
    config = HuddleConfig.load(write_config(tmp_path, data))
    assert config.peers[0].rpc_endpoint == "192.168.0.54:50100"


def test_model_resolution(tmp_path: Path) -> None:
    config = HuddleConfig.load(write_config(tmp_path, MINIMAL))
    assert config.models.resolve() == Path("/models/qwen.gguf")
    assert config.models.resolve("other.gguf") == Path("/models/other.gguf")
    assert config.models.resolve("/abs/path.gguf") == Path("/abs/path.gguf")


def test_model_resolution_without_default(tmp_path: Path) -> None:
    data = {**MINIMAL, "models": {"dir": "/models"}}
    config = HuddleConfig.load(write_config(tmp_path, data))
    with pytest.raises(ValueError, match="no model requested"):
        config.models.resolve()


def test_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUDDLE_MODEL", "override.gguf")
    config = HuddleConfig.load(write_config(tmp_path, MINIMAL))
    assert config.models.default == "override.gguf"
