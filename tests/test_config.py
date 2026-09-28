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


def test_no_config_file_is_a_valid_node() -> None:
    """A freshly installed node runs on defaults alone."""
    config = HuddleConfig.load(None)
    assert config.models.dir.name == "models"
    assert config.binaries.llama_server.name == "llama-server"


def test_installed_llamacpp_is_found(tmp_path: Path) -> None:
    from huddle import paths

    release = paths.installed_llamacpp()
    assert release is not None
    release.mkdir(parents=True)
    (release / "llama-server").write_text("")
    (release / "ggml-rpc-server").write_text("")

    config = HuddleConfig.load(None)
    assert config.binaries.llama_server == release / "llama-server"
    assert config.binaries.rpc_server == release / "ggml-rpc-server"


def test_rpc_server_is_only_taken_from_beside_llama_server(tmp_path: Path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    (build / "llama-server").write_text("")
    (build / "rpc-server").write_text("")  # upstream README's older name
    config = HuddleConfig.model_validate(
        {"binaries": {"llama_server": str(build / "llama-server")}}
    )
    assert config.binaries.rpc_server == build / "rpc-server"

    alone = tmp_path / "alone"
    alone.mkdir()
    config = HuddleConfig.model_validate(
        {"binaries": {"llama_server": str(alone / "llama-server")}}
    )
    assert config.binaries.rpc_server is None


def test_explicit_null_rpc_server_stays_unset(tmp_path: Path) -> None:
    """A node made unable to lend its GPUs on purpose must stay that way."""
    (tmp_path / "llama-server").write_text("")
    (tmp_path / "ggml-rpc-server").write_text("")
    config = HuddleConfig.model_validate(
        {"binaries": {"llama_server": str(tmp_path / "llama-server"), "rpc_server": None}}
    )
    assert config.binaries.rpc_server is None


def test_config_file_lookup_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from huddle import paths
    from huddle.config import find_config_file

    monkeypatch.chdir(tmp_path)
    assert find_config_file() is None, "no file anywhere: run on defaults"

    user = paths.default_config_file()
    user.parent.mkdir(parents=True)
    user.write_text("{}")
    assert find_config_file() == user

    (tmp_path / "huddle.yaml").write_text("{}")
    assert find_config_file() == Path("huddle.yaml"), "a checkout's own file comes first"

    monkeypatch.setenv("HUDDLE_CONFIG", "/etc/elsewhere.yaml")
    assert find_config_file() == Path("/etc/elsewhere.yaml")
    assert find_config_file(Path("given.yaml")) == Path("given.yaml")


def test_the_example_config_loads_and_matches_the_defaults() -> None:
    """huddle.example.yaml documents the defaults; it must also be valid."""
    example = HuddleConfig.load(Path(__file__).parents[1] / "huddle.example.yaml")
    defaults = HuddleConfig.load(None)
    assert example.model_dump() == defaults.model_dump()


def test_an_empty_section_means_its_defaults(tmp_path: Path) -> None:
    path = tmp_path / "huddle.yaml"
    path.write_text("binaries:\n  # llama_server: /somewhere\nmodels:\n")
    assert HuddleConfig.load(path).models.dir == HuddleConfig.load(None).models.dir
