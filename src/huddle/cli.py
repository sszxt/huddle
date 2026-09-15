"""Command-line entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from huddle import __version__
from huddle.config import HuddleConfig
from huddle.llamacpp import LlamaCppError, binary_version, list_devices

app = typer.Typer(help="Run LLMs across a cluster of Linux machines.", no_args_is_help=True)

ConfigOption = Annotated[Path, typer.Option("--config", "-f", help="Path to huddle.yaml")]
DEFAULT_CONFIG = Path("huddle.yaml")


def _load(path: Path) -> HuddleConfig:
    if not path.exists():
        typer.secho(f"config not found: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    try:
        return HuddleConfig.load(path)
    except Exception as exc:
        typer.secho(f"invalid config: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def version() -> None:
    """Print the Huddle version."""
    typer.echo(__version__)


@app.command("config")
def show_config(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Show the resolved configuration."""
    typer.echo(json.dumps(_load(config).model_dump(mode="json"), indent=2))


@app.command()
def devices(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """List the compute devices llama.cpp can see on this node."""
    settings = _load(config)
    try:
        found = list_devices(settings.binaries.llama_server)
        build = binary_version(settings.binaries.llama_server)
    except LlamaCppError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"llama.cpp {build}")
    if not found:
        typer.secho("no devices reported", fg=typer.colors.YELLOW)
        return
    for device in found:
        memory = ""
        if device.total_mib is not None:
            free = f", {device.free_mib} MiB free" if device.free_mib is not None else ""
            memory = f"  [{device.total_mib} MiB{free}]"
        typer.echo(f"  {device.id}: {device.name}{memory}")


@app.command()
def serve(
    config: ConfigOption = DEFAULT_CONFIG,
    host: Annotated[str | None, typer.Option(help="Override the API bind address")] = None,
    port: Annotated[int | None, typer.Option(help="Override the API port")] = None,
) -> None:
    """Run Huddle: the node agent and the OpenAI-compatible API."""
    import uvicorn

    from huddle.app import create_app

    settings = _load(config)
    uvicorn.run(
        create_app(settings),
        host=host or settings.api.host,
        port=port or settings.api.port,
        log_level="info",
    )


@app.command()
def agent(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Run only the node agent, for a worker node with no public API."""
    import uvicorn

    from huddle.agent.app import create_app

    settings = _load(config)
    settings.backend.autostart = False
    uvicorn.run(
        create_app(settings),
        host=settings.node.agent_host,
        port=settings.node.agent_port,
        log_level="info",
    )


@app.command("model")
def show_model(
    path: Annotated[
        Path | None, typer.Argument(help="Path to a .gguf file; defaults to the configured model")
    ] = None,
    config: ConfigOption = DEFAULT_CONFIG,
    ctx: Annotated[int, typer.Option(help="Context length for the KV cache estimate")] = 4096,
) -> None:
    """Inspect a GGUF model's metadata without loading it."""
    from huddle.gguf import GGUFError, read_gguf

    target = path
    if target is None:
        target = _load(config).models.resolve()

    try:
        info = read_gguf(target)
    except (GGUFError, OSError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    mib = 1024 * 1024
    typer.echo(f"{info.name or info.path.name}  [{info.architecture}, {info.quantization}]")
    typer.echo(f"  layers          {info.n_layers}")
    typer.echo(f"  embedding       {info.n_embd}")
    typer.echo(f"  heads           {info.n_head} ({info.n_head_kv} kv)")
    typer.echo(f"  trained ctx     {info.n_ctx_train}")
    typer.echo(f"  file size       {info.file_size / mib:,.0f} MiB")
    typer.echo(f"  tensor bytes    {info.tensor_bytes / mib:,.0f} MiB")
    typer.echo(f"  per layer       {info.mean_layer_bytes / mib:,.1f} MiB")
    typer.echo(f"  non-layer       {info.overhead_bytes / mib:,.1f} MiB")
    typer.echo(f"  kv cache @{ctx}  {info.kv_cache_bytes(ctx) / mib:,.1f} MiB")
