"""The mounted static web UI: the page, and every file it loads, is served."""

from __future__ import annotations

import re

import httpx

from huddle.web import STATIC_DIR


async def test_ui_root_serves_the_page(client: httpx.AsyncClient) -> None:
    response = await client.get("/ui/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Huddle" in response.text


async def test_ui_serves_the_app_js(client: httpx.AsyncClient) -> None:
    response = await client.get("/ui/app.js")
    assert response.status_code == 200


async def test_ui_serves_the_stylesheet(client: httpx.AsyncClient) -> None:
    response = await client.get("/ui/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


async def test_ui_page_has_the_app_shell(client: httpx.AsyncClient) -> None:
    text = (await client.get("/ui/")).text
    for element_id in ("sidebar", "sidebar-rail", "chat-container", "toaster"):
        assert f'id="{element_id}"' in text


async def test_every_local_file_the_page_references_is_served(client: httpx.AsyncClient) -> None:
    html = (STATIC_DIR / "index.html").read_text()
    paths = re.findall(r'(?:src|href)="(?!https?:)([^"]+)"', html)
    assert "app.js" in paths
    for path in paths:
        response = await client.get(f"/ui/{path}")
        assert response.status_code == 200, path


async def test_every_module_import_resolves_to_javascript(client: httpx.AsyncClient) -> None:
    # Browsers refuse an ES module served under any non-JavaScript type, and a
    # broken import path blanks the whole page with nothing on the server side.
    modules = sorted(STATIC_DIR.glob("*.js"))
    assert modules
    for module in modules:
        for target in re.findall(r'from "\./([^"]+)"', module.read_text()):
            response = await client.get(f"/ui/{target}")
            assert response.status_code == 200, f"{module.name} imports missing {target}"
            assert "javascript" in response.headers["content-type"], target
