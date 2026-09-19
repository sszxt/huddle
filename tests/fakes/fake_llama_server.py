#!/usr/bin/env python3
"""A stand-in for ``llama-server``.

The dev box has no llama.cpp, so the suite drives this instead. It mirrors the
subset of upstream's interface Huddle actually depends on, and — the real point
— records the argv it was invoked with, so tests can assert on the exact command
line Huddle would have run.

Controlled by environment variables:
  HUDDLE_FAKE_ARGV     file to write the received argv to, as JSON
  HUDDLE_FAKE_DEVICES  newline-separated device lines for --list-devices
  HUDDLE_FAKE_DELAY    seconds to wait before binding, to exercise readiness
  HUDDLE_FAKE_MAX_LOCAL_LAYERS
                       run out of device memory, with llama.cpp's real Vulkan
                       error lines, when a single-node launch offloads more
                       entries (-ngl: layers plus the output head) than this
  HUDDLE_FAKE_OOM_DEVICE
                       device named in that error (default Vulkan0)
  HUDDLE_FAKE_VERSION  what --version reports, to simulate a mismatched build
  HUDDLE_FAKE_LOG_FLOOD
                       lines of filler to print after startup, the way -lv 5
                       buries everything useful under noise
  HUDDLE_FAKE_TOKENS_PER_SEC
                       emit a canned "... tokens per second)" line after
                       startup, to exercise BackendStatus.tokens_per_sec

At -lv 5 it prints "layer N assigned to device X" lines using the rule measured
on real hardware: entries 0..n_layer, the first n_layer+1-ngl on CPU, the rest
split contiguously by --tensor-split in device order, RPC devices first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The GGUF reader is plain stdlib, so the fake can use the real one to learn how
# many layers the model has, the same way llama.cpp does.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from huddle.gguf import GGUFError, read_gguf

# Same shape as the real build's output, at the pinned commit, so version checks
# see what they would see on a node built from llamacpp.pin.
FAKE_VERSION = "version: 0.4.1-dev (build 1, commit 4c9233c)"

DEFAULT_DEVICES = (
    "Available devices:\n"
    "  Vulkan0: NVIDIA GeForce RTX 5070 (12227 MiB, 11041 MiB free)\n"
    "  Vulkan1: Intel(R) Graphics (RPL-S) (8192 MiB, 7900 MiB free)\n"
)

CHAT_REPLY = "huddle fake response"


def build_parser() -> argparse.ArgumentParser:
    """Accept the flag surface Huddle builds, so a wrong flag fails loudly."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-m", "--model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("-c", "--ctx-size", type=int, default=4096)
    parser.add_argument("-np", "--parallel", type=int, default=1)
    parser.add_argument("-ngl", "--gpu-layers", "--n-gpu-layers", dest="gpu_layers", default="0")
    parser.add_argument("--rpc")
    parser.add_argument("--tensor-split")
    parser.add_argument("--alias")
    parser.add_argument("--api-key")
    parser.add_argument("--log-file")
    parser.add_argument("-lv", "--verbosity", type=int, default=3)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--version", action="store_true")
    return parser


class Handler(BaseHTTPRequestHandler):
    args: argparse.Namespace

    def log_message(self, *_: object) -> None:  # keep test output readable
        pass

    def _send(self, code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/props":
            self._send(200, {"model_path": self.args.model, "n_ctx": self.args.ctx_size})
        elif self.path == "/v1/models":
            name = self.args.alias or (self.args.model or "fake-model")
            self._send(200, {"object": "list", "data": [{"id": name, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path not in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings"):
            self._send(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return

        if self.path == "/v1/embeddings":
            self._send(
                200,
                {
                    "object": "list",
                    "model": request.get("model", "fake-model"),
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                },
            )
        elif request.get("stream"):
            self._stream_reply(request)
        else:
            self._send(
                200,
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "model": request.get("model", "fake-model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": CHAT_REPLY},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

    def _stream_reply(self, request: dict[str, object]) -> None:
        """Emit SSE chunks, so proxy buffering bugs surface in tests."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        model = request.get("model", "fake-model")
        for token in CHAT_REPLY.split():
            chunk = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {"content": token + " "}}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def emit(args: argparse.Namespace, line: str) -> None:
    """Print a log line, and append it to --log-file as llama.cpp would."""
    print(line, flush=True)
    if args.log_file:
        with open(args.log_file, "a") as handle:
            handle.write(line + "\n")


def local_device_ids() -> list[str]:
    listing = os.environ.get("HUDDLE_FAKE_DEVICES", DEFAULT_DEVICES)
    return [
        line.split(":", 1)[0].strip()
        for line in listing.splitlines()
        if ":" in line and not line.lower().startswith("available")
    ]


def placement_lines(args: argparse.Namespace) -> list[str]:
    """Layer assignment output, following llama.cpp's layer-split rule."""
    try:
        n_layer = read_gguf(args.model).n_layers if args.model else 0
    except (GGUFError, OSError):
        return []

    total = n_layer + 1  # llama.cpp counts the output layer as one more entry
    try:
        requested = int(args.gpu_layers)
    except ValueError:
        requested = total
    offloaded = max(0, min(requested, total))
    first_gpu = total - offloaded

    local = local_device_ids()
    weights = [float(w) for w in args.tensor_split.split(",")] if args.tensor_split else []
    if not weights:
        names, weights = local[:1], [1.0]
    else:
        remote = len(weights) - len(local) if args.rpc else 0
        names = [f"RPC{i}" for i in range(max(remote, 0))] + local
        names = names[: len(weights)]

    total_weight = sum(weights) or 1.0
    cumulative, running = [], 0.0
    for weight in weights:
        running += weight / total_weight
        cumulative.append(running)

    lines = []
    for layer in range(total):
        device = "CPU"
        if layer >= first_gpu and offloaded:
            fraction = (layer - first_gpu) / offloaded
            index = next((k for k, c in enumerate(cumulative) if fraction < c), len(names) - 1)
            device = names[index]
        lines.append(
            f"0.00.000.000 D load_tensors: layer {layer:3} assigned to device {device}, is_swa = 0"
        )
    return lines


def out_of_memory(args: argparse.Namespace) -> bool:
    """Whether this launch should fail the way an over-full GPU does.

    Only single-node launches: with --rpc the fake cannot tell which split
    position belongs to which device, since peers expose several each.
    """
    limit = os.environ.get("HUDDLE_FAKE_MAX_LOCAL_LAYERS")
    if limit is None or args.rpc:
        return False
    try:
        return int(args.gpu_layers) > int(limit)
    except ValueError:
        return True  # "all" or "auto" asks for everything


def main() -> int:
    argv_file = os.environ.get("HUDDLE_FAKE_ARGV")
    if argv_file:
        with open(argv_file, "w") as handle:
            json.dump(sys.argv, handle)

    args, _unknown = build_parser().parse_known_args()

    if args.version:
        print(os.environ.get("HUDDLE_FAKE_VERSION", FAKE_VERSION), file=sys.stderr)
        return 0

    if args.list_devices:
        sys.stdout.write(os.environ.get("HUDDLE_FAKE_DEVICES", DEFAULT_DEVICES))
        return 0

    delay = float(os.environ.get("HUDDLE_FAKE_DELAY", "0"))
    if delay:
        time.sleep(delay)

    if out_of_memory(args):
        device = os.environ.get("HUDDLE_FAKE_OOM_DEVICE", "Vulkan0")
        # Verbatim shape of the real failure captured on omarchy.
        emit(args, "ggml_vulkan: Device memory allocation of size 1071374336 failed.")
        emit(args, "ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory")
        emit(args, f"E alloc_tensor_range: failed to allocate {device} buffer of size 1071374336")
        emit(args, f"E llama_model_load: error loading model: unable to allocate {device} buffer")
        emit(args, "E srv  llama_server: exiting due to model loading error")
        return 1

    Handler.args = args
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if args.verbosity >= 5:
        # llama.cpp prints a fitting dry run and then the real load.
        for _ in range(2):
            for line in placement_lines(args):
                emit(args, line)
    for i in range(int(os.environ.get("HUDDLE_FAKE_LOG_FLOOD", "0"))):
        emit(args, f"0.00.000.000 D arg_name_suffix: '' ({i})")
    tps = os.environ.get("HUDDLE_FAKE_TOKENS_PER_SEC")
    if tps:
        emit(
            args,
            f"eval time =   123.45 ms /    10 runs (  12.34 ms per token, {tps} tokens per second)",
        )
    emit(args, f"fake llama-server listening on {args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
