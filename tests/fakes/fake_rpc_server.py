#!/usr/bin/env python3
"""A stand-in for ``rpc-server``.

Speaks no RPC protocol — it only parses the flags Huddle passes and holds the
port open, which is enough to test bind addresses, port allocation, readiness
and shutdown without llama.cpp installed.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time


def main() -> int:
    argv_file = os.environ.get("HUDDLE_FAKE_ARGV")
    if argv_file:
        with open(argv_file, "w") as handle:
            json.dump(sys.argv, handle)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-H", "--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=50052)
    parser.add_argument("-c", "--cache", action="store_true")
    parser.add_argument("-d", "--device")
    parser.add_argument("-t", "--threads", type=int)
    args, _unknown = parser.parse_known_args()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, args.port))
    listener.listen(8)
    print(f"fake rpc-server listening on {args.host}:{args.port}", flush=True)

    while True:  # held open until supervised shutdown
        time.sleep(3600)


if __name__ == "__main__":
    raise SystemExit(main())
