"""The llama.cpp build every node runs, as recorded in ``llamacpp.pin``.

The pin ships inside the package, so an installed Huddle knows which release
to fetch and a node can tell whether its binaries match its peers'. The file
stays shell-sourceable because ``scripts/bootstrap-node.sh`` reads it too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PIN_FILE = Path(__file__).with_name("llamacpp.pin")

# Upstream's own names for its release architectures.
ARCHES = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}


@dataclass(frozen=True)
class Pin:
    repo: str
    ref: str
    release: str | None = None
    # Asset suffix ("vulkan-x64") -> sha256 of the release tarball.
    sha256: dict[str, str] = field(default_factory=dict)

    def asset(self, arch: str, backend: str = "vulkan") -> str:
        """Upstream's release tarball name, e.g. ``llama-b10976-bin-ubuntu-vulkan-x64.tar.gz``."""
        return f"llama-{self.release}-bin-ubuntu-{backend}-{arch}.tar.gz"

    def asset_url(self, arch: str, backend: str = "vulkan") -> str:
        return f"{self.repo}/releases/download/{self.release}/{self.asset(arch, backend)}"

    def digest(self, arch: str, backend: str = "vulkan") -> str | None:
        return self.sha256.get(f"{backend}-{arch}")


def parse_pin(text: str) -> Pin | None:
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if value.strip():
            values[key.strip()] = value.strip()
    if "LLAMACPP_REF" not in values:
        return None
    digests = {
        key.removeprefix("LLAMACPP_SHA256_").lower().replace("_", "-"): value
        for key, value in values.items()
        if key.startswith("LLAMACPP_SHA256_")
    }
    return Pin(
        repo=values.get("LLAMACPP_REPO", "https://github.com/ggml-org/llama.cpp"),
        ref=values["LLAMACPP_REF"],
        release=values.get("LLAMACPP_RELEASE"),
        sha256=digests,
    )


def load_pin(path: Path = PIN_FILE) -> Pin | None:
    try:
        return parse_pin(path.read_text())
    except OSError:
        return None


__all__ = ["ARCHES", "PIN_FILE", "Pin", "load_pin", "parse_pin"]
