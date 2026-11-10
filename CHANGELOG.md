# Changelog

All notable changes to Huddle. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Entries say what was checked and where: "unit-tested" means against the fake
llama.cpp binaries in `tests/fakes/`, which shows the code does what it was
told and nothing more.

## [Unreleased]

### Added
- Web UI accounts: the first account is the admin, later sign-ups wait for
  approval, and each user's chat history is kept on the PC serving the page
  (SQLite, scrypt password hashes). Unit-tested; not yet run on a real node.
- LICENSE (MIT), packaging metadata, issue templates.
- `huddle admin-code`: the first account, when created from another device,
  needs a one-time setup code (logged at start-up); from the PC itself it needs
  none. Wrong guesses count against the sign-in rate limit.
- Upstream failures reach API callers as OpenAI-shaped errors
  (`error.message/type/code`, and `detail` for the web UI): 503 with
  `Retry-After` while the cluster restarts its model server, 502 when it cannot
  be reached, 504 on a timeout. A stream that dies part-way ends with an error
  event and `[DONE]` instead of silently stopping.

### Fixed
- A model named in a request (`/cluster/start`, `/cluster/model`,
  `/cluster/plan`, `/agent/backend/start`) can no longer be an absolute path or
  contain `..`; it must be a plain `.gguf` file name. Download requests are
  validated the same way (`owner/name` repos, relative `.gguf` paths only).
- The first sign-up could be taken by anyone on the network (see
  `huddle admin-code` above).
- A dead model server surfaced as a bare 500.
- A corrupt or hostile `.gguf` header could make Huddle raise an unexpected
  error or try to allocate gigabytes (a lying length, a layer count of a
  billion, arrays nested thousands deep). It is now refused with a plain
  "not a readable GGUF" error. Found by fuzzing the parser.
- `huddle plan` now shows the placement a start would use: it applies
  `planner.reserve_mib` and drops peers that were given no layers.
- A GPU reporting 0 MiB free was planned as if its whole memory were free.

## [0.2.0] - 2026-09-28

### Added
- One-line installer (`install.sh`) and `huddle setup`: installs the pinned
  prebuilt llama.cpp release (checksum-verified), a systemd service and
  subnet-only firewall rules. No compiling.
- Zero-config clustering: every PC is equal, found over mDNS, and whichever
  loads a model becomes the head. Workers are leased and handed over between
  PCs.
- Web UI first-run screen with a catalog of models sized to the cluster, and
  models listed from every PC.
- Cluster page showing each node's hardware, round-trip time and which layers
  sit on which device.

### Changed
- llama.cpp is pinned to upstream release b10976 instead of a source build.

## [0.1.0] - 2026-09-15

The version from the first commit; nothing was tagged. Everything built up to
the installer work (2026-09-15 to 2026-09-26) is listed here.

### Added
- Single-node OpenAI-compatible API in front of `llama-server`.
- Multi-node layer split over llama.cpp RPC: GGUF header parser, placement
  planner, supervised restart, model switching, systemd units.
- mDNS peer discovery, out-of-memory replanning, peer rejoin.
- `huddle doctor`, `huddle tui`, Hugging Face model search and download, and
  a chat web UI.

Measured on two Ubuntu nodes with one RTX 5070 each (a past run, not
repeated since the hardware became unavailable): a 32B Q4_K_M model that fits
on neither GPU served fully offloaded across both at 27.7 tok/s generation.
