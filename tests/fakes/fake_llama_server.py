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
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FAKE_VERSION = "version: 9999 (fakebuild)"

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


def main() -> int:
    argv_file = os.environ.get("HUDDLE_FAKE_ARGV")
    if argv_file:
        with open(argv_file, "w") as handle:
            json.dump(sys.argv, handle)

    args, _unknown = build_parser().parse_known_args()

    if args.version:
        print(FAKE_VERSION, file=sys.stderr)
        return 0

    if args.list_devices:
        sys.stdout.write(os.environ.get("HUDDLE_FAKE_DEVICES", DEFAULT_DEVICES))
        return 0

    delay = float(os.environ.get("HUDDLE_FAKE_DELAY", "0"))
    if delay:
        time.sleep(delay)

    Handler.args = args
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"fake llama-server listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
