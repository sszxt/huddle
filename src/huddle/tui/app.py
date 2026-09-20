"""The interactive terminal dashboard: `huddle tui`.

Modeled on exo's own terminal visualizer (`exo/viz/topology_viz.py`): a plain
`rich.live.Live` loop that rebuilds a renderable from the latest
`ClusterSnapshot` and hands it to `Live.update()` — see `render.py` for what
gets built. No placement, peer-resolution, or orchestration logic lives here;
every control is a thin call into `ClusterPoller`'s wrappers over the
existing `/cluster/*` endpoints — see `poller.py` for where the data and the
controls actually come from.

Keyboard input reads raw single keypresses directly (POSIX `termios`/`tty`),
rather than pulling in a widget framework for five keybindings. This matches
every other Linux-only assumption already in this project (e.g. `hardware.py`
reads `/proc/meminfo` unconditionally).
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import termios
import tty
from collections.abc import Iterator
from typing import Any

import httpx
from rich.console import Console
from rich.live import Live

from huddle.config import HuddleConfig
from huddle.tui.poller import ClusterPoller, ClusterSnapshot
from huddle.tui.render import frame


def _detail(exc: httpx.HTTPStatusError) -> str:
    try:
        payload = exc.response.json()
        if isinstance(payload, dict) and "detail" in payload:
            return str(payload["detail"])
    except ValueError:
        pass
    return exc.response.text or str(exc)


class HuddleTUI:
    """Owns the terminal: polls the cluster, redraws, and reads keypresses."""

    def __init__(self, config: HuddleConfig, *, poll_interval: float | None = None) -> None:
        self.config = config
        self.poll_interval = poll_interval or config.supervisor.watch_interval
        self.poller = ClusterPoller(config)
        self.snapshot: ClusterSnapshot | None = None
        self.message: str | None = None
        self.console = Console()
        self._live: Live | None = None
        self._cooked_settings: list[Any] | None = None

    async def run(self) -> None:
        with Live(
            frame(None, self.config), console=self.console, screen=True, auto_refresh=False
        ) as live:
            self._live = live
            poll_task = asyncio.create_task(self._poll_loop())
            try:
                await self._input_loop()
            finally:
                poll_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poll_task
                await self.poller.aclose()

    def _redraw(self) -> None:
        if self._live is not None:
            self._live.update(frame(self.snapshot, self.config, message=self.message), refresh=True)

    async def _poll_loop(self) -> None:
        while True:
            await self._refresh()
            await asyncio.sleep(self.poll_interval)

    async def _refresh(self) -> None:
        try:
            self.snapshot = await self.poller.poll()
            self.message = None
        except Exception as exc:  # a poll failure must never crash the dashboard
            self.message = f"poll failed: {exc}"
        self._redraw()

    # -- keyboard ------------------------------------------------------

    async def _input_loop(self) -> None:
        fd = sys.stdin.fileno()
        self._cooked_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        loop = asyncio.get_running_loop()
        try:
            while True:
                key = await loop.run_in_executor(None, sys.stdin.read, 1)
                if key == "q":
                    return
                if key == "s":
                    await self._start_cluster()
                elif key == "x":
                    await self._stop_cluster_confirmed()
                elif key == "m":
                    await self._switch_model_prompt()
                elif key == "r":
                    await self._refresh()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._cooked_settings)

    @contextlib.contextmanager
    def _paused_for_input(self) -> Iterator[None]:
        """Hand the terminal back to a plain `input()` prompt, then resume.

        The keyboard loop's cbreak mode disables line editing and echo, which
        `input()` needs; this restores the terminal's original ("cooked")
        settings for the duration of one prompt, matching the pattern Rich's
        own docs recommend for interactive input alongside a `Live` display.
        """
        live = self._live
        assert live is not None
        assert self._cooked_settings is not None
        fd = sys.stdin.fileno()
        live.stop()
        termios.tcsetattr(fd, termios.TCSADRAIN, self._cooked_settings)
        try:
            yield
        finally:
            tty.setcbreak(fd)
            live.start(refresh=True)

    # -- actions ---------------------------------------------------------

    async def _start_cluster(self) -> None:
        try:
            await self.poller.start_cluster()
        except httpx.HTTPStatusError as exc:
            self.message = f"start failed: {_detail(exc)}"
            self._redraw()
        except httpx.HTTPError as exc:
            self.message = f"start failed: {exc}"
            self._redraw()
        else:
            await self._refresh()

    async def _stop_cluster_confirmed(self) -> None:
        """The interactive 'x' path: ask, then delegate to `_stop_cluster`."""
        with self._paused_for_input():
            answer = input("Stop the cluster? [y/N] ").strip().lower()
        if answer == "y":
            await self._stop_cluster()

    async def _stop_cluster(self) -> None:
        try:
            await self.poller.stop_cluster()
        except httpx.HTTPError as exc:
            self.message = f"stop failed: {exc}"
            self._redraw()
        else:
            await self._refresh()

    async def _switch_model_prompt(self) -> None:
        """The interactive 'm' path: ask, then delegate to `_switch_model`."""
        models = self.snapshot.available_models if self.snapshot else []
        if not models:
            self.message = "no models available"
            self._redraw()
            return
        with self._paused_for_input():
            print("Available models:")
            for i, name in enumerate(models, 1):
                print(f"  {i}. {name}")
            choice = input("Switch to # (blank to cancel): ").strip()
        if not choice:
            return
        try:
            model = models[int(choice) - 1]
        except (ValueError, IndexError):
            self.message = f"invalid choice: {choice!r}"
            self._redraw()
            return
        await self._switch_model(model)

    async def _switch_model(self, model: str) -> None:
        try:
            await self.poller.switch_model(model)
        except httpx.HTTPStatusError as exc:
            self.message = f"switch failed: {_detail(exc)}"
            self._redraw()
        except httpx.HTTPError as exc:
            self.message = f"switch failed: {exc}"
            self._redraw()
        else:
            await self._refresh()


def run_tui(config: HuddleConfig, *, poll_interval: float | None = None) -> None:
    asyncio.run(HuddleTUI(config, poll_interval=poll_interval).run())


__all__ = ["HuddleTUI", "run_tui"]
