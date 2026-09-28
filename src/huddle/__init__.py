"""Huddle — run GGUF models across a cluster of Linux machines."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("huddle")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"
