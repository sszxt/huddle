"""The first-run model list: what it offers, and whether each model is said to fit."""

from __future__ import annotations

import httpx

from huddle.app import create_app
from huddle.catalog import CATALOG, Capacity, capacity, fit_for, need_mib
from huddle.config import HuddleConfig
from huddle.hardware import DeviceInfo, NodeHardware

GIB = 1024**3


def node(*devices: DeviceInfo, ram_mib: int = 60_000) -> NodeHardware:
    return NodeHardware(name="n", devices=list(devices), ram_available_mib=ram_mib)


def test_catalog_entries_are_single_gguf_files() -> None:
    assert len({m.file for m in CATALOG}) == len(CATALOG)
    for model in CATALOG:
        assert model.file.endswith(".gguf") and "/" not in model.file
        assert "-of-" not in model.file, "split models need every shard; offer none"
        assert model.size > 100 * 1024**2


def test_capacity_counts_discrete_gpus_only_after_headroom() -> None:
    rtx = DeviceInfo(id="Vulkan0", name="RTX 5070", total_mib=12_473, free_mib=12_000)
    igpu = DeviceInfo(id="Vulkan1", name="UHD 770", free_mib=47_000, unified_memory=True)
    head = node(rtx, igpu, ram_mib=50_000)
    room = capacity([head, node(rtx)], head, headroom=0.15)
    assert room.nodes == 2
    assert room.gpu_mib == 2 * int(12_000 * 0.85), "an iGPU's figure is system RAM"
    assert room.head_ram_mib == 50_000


def test_fit_labels() -> None:
    two_gpus = Capacity(nodes=2, gpu_mib=20_400, head_ram_mib=50_000)
    assert fit_for(9 * GIB, two_gpus) == "gpu"
    assert fit_for(int(19.85 * GIB), two_gpus) == "partial", "the 32B needed both GPUs and more"
    assert fit_for(int(42.5 * GIB), two_gpus) == "partial"
    assert fit_for(200 * GIB, two_gpus) == "too_large"
    assert need_mib(GIB) > 1024, "weights are not the whole cost"


async def test_catalog_route_labels_models_for_this_cluster(
    huddle_config: HuddleConfig,
) -> None:
    huddle_config.backend.autostart = False
    (huddle_config.models.dir / CATALOG[0].file).write_bytes(b"GGUF")
    app = create_app(huddle_config)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://huddle.test") as http,
    ):
        body = (await http.get("/cluster/catalog")).json()

    assert body["capacity"]["nodes"] == 1
    assert body["capacity"]["gpu_mib"] > 0
    models = {m["file"]: m for m in body["models"]}
    assert set(models) == {m.file for m in CATALOG}
    assert models[CATALOG[0].file]["on_node"] == "testnode", "already downloaded here"
    assert {m["fit"] for m in body["models"]} <= {"gpu", "partial", "too_large"}


async def test_root_opens_the_web_ui(huddle_config: HuddleConfig) -> None:
    huddle_config.backend.autostart = False
    app = create_app(huddle_config)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://huddle.test") as http,
    ):
        response = await http.get("/")
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"
