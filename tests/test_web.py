"""The mounted static model-manager page — just confirms the mount works."""

from __future__ import annotations

import httpx


async def test_ui_root_serves_the_page(client: httpx.AsyncClient) -> None:
    response = await client.get("/ui/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "HUDDLE" in response.text


async def test_ui_serves_the_app_js(client: httpx.AsyncClient) -> None:
    response = await client.get("/ui/app.js")
    assert response.status_code == 200
