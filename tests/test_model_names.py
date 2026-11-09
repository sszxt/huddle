"""A model named in a request must be a plain file in the models directory.

Anyone who can reach a node's port can send a model name. Without a check, an
absolute path or ``..`` walks out of the models directory, so these pin the
boundary: the network routes refuse such names, while the config file and the
local CLI (which are trusted) keep their freedom to point anywhere.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from huddle import catalog
from huddle.config import InvalidModelName, ModelsConfig
from huddle.coordinator.app import DownloadRequest

BAD_NAMES = [
    "../escape.gguf",
    "../../etc/passwd",
    "/etc/passwd",
    "/abs/path/model.gguf",
    "sub/dir/model.gguf",
    "..\\windows.gguf",
    "notes.txt",
    "model.gguf.sh",
    "..",
    ".",
    "tiny.gguf\0",
]


# -- the check itself -----------------------------------------------------------------


@pytest.mark.parametrize("name", BAD_NAMES)
def test_a_name_that_reaches_outside_the_directory_is_refused(name: str) -> None:
    with pytest.raises(InvalidModelName):
        ModelsConfig().check_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "tiny.gguf",
        "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "model-00001-of-00003.gguf",  # a split model is loaded by its first shard
        "UPPER.GGUF",
    ],
)
def test_a_plain_gguf_name_is_accepted(name: str) -> None:
    ModelsConfig().check_name(name)


def test_no_name_means_the_default_and_is_accepted() -> None:
    ModelsConfig().check_name(None)
    ModelsConfig().check_name("")


def test_the_configured_default_is_accepted_as_written(tmp_path: Path) -> None:
    """The config file is trusted, so a default may be an absolute path."""
    default = str(tmp_path / "elsewhere" / "big.gguf")
    config = ModelsConfig(dir=tmp_path / "models", default=default)

    config.check_name(default)
    with pytest.raises(InvalidModelName):
        config.check_name(str(tmp_path / "elsewhere" / "other.gguf"))


def test_resolve_stays_permissive_for_trusted_callers(tmp_path: Path) -> None:
    config = ModelsConfig(dir=tmp_path / "models")
    absolute = tmp_path / "elsewhere" / "big.gguf"
    assert config.resolve(str(absolute)) == absolute
    assert config.resolve("a.gguf") == tmp_path / "models" / "a.gguf"


# -- the routes -----------------------------------------------------------------------


@pytest.mark.parametrize("name", [n for n in BAD_NAMES if "\0" not in n])
async def test_cluster_start_refuses_a_bad_name(client: httpx.AsyncClient, name: str) -> None:
    response = await client.post("/cluster/start", json={"model": name})
    assert response.status_code == 422, response.text
    assert "model file name" in response.json()["detail"]


async def test_cluster_model_switch_refuses_a_bad_name(client: httpx.AsyncClient) -> None:
    response = await client.post("/cluster/model", json={"model": "../../etc/passwd"})
    assert response.status_code == 422


async def test_cluster_plan_refuses_a_bad_name(client: httpx.AsyncClient) -> None:
    response = await client.get("/cluster/plan", params={"model": "/etc/passwd"})
    assert response.status_code == 422


async def test_agent_backend_start_refuses_a_bad_name(client: httpx.AsyncClient) -> None:
    response = await client.post("/agent/backend/start", json={"model": "../x.gguf"})
    assert response.status_code == 422


async def test_a_good_name_still_plans(client: httpx.AsyncClient) -> None:
    response = await client.get("/cluster/plan", params={"model": "tiny.gguf"})
    assert response.status_code == 200, response.text


async def test_a_good_but_missing_name_is_still_a_conflict_not_a_validation_error(
    client: httpx.AsyncClient,
) -> None:
    """Valid names that do not exist keep their old answer."""
    response = await client.post("/cluster/model", json={"model": "nope.gguf"})
    assert response.status_code == 409


# -- downloads ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "../x.gguf",
        "a/../../x.gguf",
        "/abs/model.gguf",
        "\\abs\\model.gguf",
        "model.bin",
        "model.gguf\0",
    ],
)
def test_a_download_filename_cannot_climb_out_or_be_another_type(filename: str) -> None:
    with pytest.raises(ValidationError):
        DownloadRequest(repo_id="owner/name", filename=filename)


@pytest.mark.parametrize("repo_id", ["", "noslash", "a/b/c", "../x", "owner/na me", "/owner/name"])
def test_a_download_repo_id_must_look_like_owner_slash_name(repo_id: str) -> None:
    with pytest.raises(ValidationError):
        DownloadRequest(repo_id=repo_id, filename="model.gguf")


@pytest.mark.parametrize(
    ("repo_id", "filename"),
    [
        ("bartowski/Qwen2.5-32B-Instruct-GGUF", "Qwen2.5-32B-Instruct-Q4_K_M.gguf"),
        ("owner/name", "Q4_K_M/model-00001-of-00002.gguf"),
    ],
)
def test_ordinary_downloads_are_accepted(repo_id: str, filename: str) -> None:
    DownloadRequest(repo_id=repo_id, filename=filename)


def test_every_catalog_entry_passes_the_download_check() -> None:
    """Our own recommendations must not be refused by our own validator."""
    for model in catalog.CATALOG:
        DownloadRequest(repo_id=model.repo_id, filename=model.file)


async def test_the_download_route_refuses_a_bad_request_before_touching_the_network(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/cluster/models/download", json={"repo_id": "owner/name", "filename": "../x.gguf"}
    )
    assert response.status_code == 422
