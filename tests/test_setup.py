"""`huddle setup`: every step, with nothing actually installed.

The commands setup would run are recorded rather than run, so these tests say
what a node would be told to do. Whether apt, systemd and ufw then do the right
thing is only shown by running the installer on a real machine.
"""

from __future__ import annotations

import hashlib
import io
import subprocess
import tarfile
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from huddle import paths
from huddle.pin import PIN_FILE, Pin, load_pin, parse_pin
from huddle.setup import (
    UNIT_MARKER,
    Installed,
    Runner,
    SetupError,
    build_matches,
    config_text,
    download,
    extract_release,
    firewall_commands,
    install_llamacpp,
    install_service,
    machine_arch,
    missing_libraries,
    packages_for,
    run_setup,
    run_uninstall,
    unit_text,
)

VERSION = "version: 0.4.1-dev (build 10976, commit 987498f45)"
PIN = Pin(
    repo="https://github.com/ggml-org/llama.cpp",
    ref="987498f4592a76897863cf53711dce38380c082b",
    release="b10976",
    sha256={"vulkan-x64": "0" * 64},
)


class FakeRunner(Runner):
    """Records commands instead of running them."""

    def __init__(self, stdout: dict[str, str] | None = None) -> None:
        super().__init__()
        self.calls: list[tuple[list[str], bool, str | None]] = []
        self.stdout = stdout or {}

    def run(
        self,
        argv: Sequence[str],
        *,
        sudo: bool = False,
        input: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), sudo, input))
        return subprocess.CompletedProcess(list(argv), 0, self.stdout.get(argv[0], ""), "")

    def commands(self) -> list[str]:
        return [" ".join(argv) for argv, _, _ in self.calls]


def fake_release(path: Path, *, tag: str = "b10976", version: str = VERSION) -> bytes:
    """A tarball shaped like upstream's: one top directory, scripts for binaries."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, body, mode in (
            ("llama-server", f"#!/bin/sh\necho '{version}'\n", 0o755),
            ("ggml-rpc-server", "#!/bin/sh\nexit 0\n", 0o755),
            ("libggml.so.0", "not really a library", 0o644),
        ):
            data = body.encode()
            info = tarfile.TarInfo(f"llama-{tag}/{name}")
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo(f"llama-{tag}/libggml.so")
        link.type, link.linkname = tarfile.SYMTYPE, "libggml.so.0"
        tar.addfile(link)
    path.write_bytes(buffer.getvalue())
    return buffer.getvalue()


# -- the pin -----------------------------------------------------------------------


def test_shipped_pin_names_a_release_with_checksums() -> None:
    """The pin travels inside the package, so an installed Huddle can read it."""
    pin = load_pin(PIN_FILE)
    assert pin is not None and pin.release is not None
    assert pin.ref.startswith("987498f")
    assert len(pin.digest("x64") or "") == 64
    assert len(pin.digest("arm64") or "") == 64
    assert pin.asset_url("x64") == (
        "https://github.com/ggml-org/llama.cpp/releases/download/"
        f"{pin.release}/llama-{pin.release}-bin-ubuntu-vulkan-x64.tar.gz"
    )


def test_pin_without_a_ref_is_rejected() -> None:
    assert parse_pin("LLAMACPP_RELEASE=b1\n") is None


@pytest.mark.parametrize(
    ("machine", "arch"), [("x86_64", "x64"), ("AMD64", "x64"), ("aarch64", "arm64")]
)
def test_machine_arch(machine: str, arch: str) -> None:
    assert machine_arch(machine) == arch


def test_unsupported_arch_says_how_to_build() -> None:
    with pytest.raises(SetupError, match=r"bootstrap-node\.sh"):
        machine_arch("riscv64")


def test_build_matches_prefixes_either_way() -> None:
    assert build_matches(VERSION, PIN)
    assert build_matches("version: 0.4.1-dev (build 1, commit 987498f)", PIN)
    assert not build_matches("version: 0.4.1-dev (build 10975, commit 4c9233c03)", PIN)
    assert not build_matches("unknown", PIN)


# -- download and unpack -------------------------------------------------------------


def _client(body: bytes) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=body))
    )


def test_download_keeps_a_file_whose_checksum_matches(tmp_path: Path) -> None:
    body = b"release bytes"
    dest = tmp_path / "pkg.tar.gz"
    with _client(body) as client:
        download(
            "https://example.test/pkg",
            dest,
            hashlib.sha256(body).hexdigest(),
            echo=lambda _line: None,
            client=client,
        )
    assert dest.read_bytes() == body


def test_download_refuses_a_file_whose_checksum_differs(tmp_path: Path) -> None:
    dest = tmp_path / "pkg.tar.gz"
    with _client(b"tampered") as client, pytest.raises(SetupError, match="checksum mismatch"):
        download("https://example.test/pkg", dest, "0" * 64, echo=print, client=client)
    assert not dest.exists(), "a file that failed verification must not be left behind"


def test_extract_release_strips_the_top_directory(tmp_path: Path) -> None:
    archive = tmp_path / "pkg.tar.gz"
    fake_release(archive)
    target = tmp_path / "root" / "b10976"
    target.parent.mkdir()
    extract_release(archive, target)
    assert (target / "llama-server").is_file()
    assert (target / "llama-server").stat().st_mode & 0o111, "binaries stay executable"
    assert (target / "libggml.so").is_symlink()
    assert not target.with_name("b10976.partial").exists()


def test_extract_release_refuses_paths_outside_the_target(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escaped")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    target = tmp_path / "root" / "b1"
    target.parent.mkdir()
    with pytest.raises(SetupError):
        extract_release(archive, target)
    assert not (tmp_path / "root" / "escaped").exists()
    assert not (tmp_path / "escaped").exists()


def test_install_downloads_once_then_reuses(tmp_path: Path) -> None:
    fetched: list[str] = []

    def fetch(url: str, dest: Path, sha256: str) -> None:
        fetched.append(url)
        fake_release(dest)

    root = tmp_path / "llama.cpp"
    (root / "b9999").mkdir(parents=True)  # an older release, no longer wanted

    server = install_llamacpp(PIN, root=root, arch="x64", echo=lambda _l: None, fetch=fetch)
    assert server == root / "b10976" / "llama-server"
    assert len(fetched) == 1 and fetched[0].endswith("llama-b10976-bin-ubuntu-vulkan-x64.tar.gz")
    assert not (root / "b9999").exists(), "old releases are removed"
    assert not list(root.glob("*.download"))

    lines: list[str] = []
    install_llamacpp(PIN, root=root, arch="x64", echo=lines.append, fetch=fetch)
    assert len(fetched) == 1, "an install that already matches the pin is not re-downloaded"
    assert "already installed" in lines[0]


def test_install_replaces_a_build_that_is_off_the_pin(tmp_path: Path) -> None:
    root = tmp_path / "llama.cpp"
    fake_release(tmp_path / "old.tar.gz", version="version: x (build 1, commit deadbeef)")
    (root).mkdir()
    extract_release(tmp_path / "old.tar.gz", root / "b10976")

    fetched: list[str] = []

    def fetch(url: str, dest: Path, sha256: str) -> None:
        fetched.append(url)
        fake_release(dest)

    install_llamacpp(PIN, root=root, arch="x64", echo=lambda _l: None, fetch=fetch)
    assert fetched, "a binary reporting another commit is replaced"


def test_install_needs_a_checksum_for_the_arch(tmp_path: Path) -> None:
    with pytest.raises(SetupError, match="no checksum for the arm64"):
        install_llamacpp(PIN, root=tmp_path, arch="arm64", echo=lambda _l: None)


# -- libraries -------------------------------------------------------------------------


def test_missing_libraries_from_ldd() -> None:
    output = (
        "\tlinux-vdso.so.1 (0x00007ffd)\n"
        "\tlibvulkan.so.1 => not found\n"
        "\tlibgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007f)\n"
        "\tlibssl.so.3 => not found\n"
    )
    assert missing_libraries(output) == {"libvulkan.so.1", "libssl.so.3"}


def test_packages_for_missing_libraries() -> None:
    packages, unknown = packages_for({"libvulkan.so.1", "libssl.so.3", "libcrypto.so.3"}, "apt")
    assert packages == ["libssl3", "libvulkan1"], "one package covers both OpenSSL sonames"
    assert unknown == []
    assert packages_for({"libmystery.so.9"}, "dnf") == ([], ["libmystery.so.9"])


# -- config -----------------------------------------------------------------------------


def test_config_text_is_valid_yaml_with_the_choices(tmp_path: Path) -> None:
    import yaml

    text = config_text(cluster="lab 2", models_dir=Path("/srv/models"))
    data = yaml.safe_load(text)
    assert data == {"discovery": {"cluster": "lab 2"}, "models": {"dir": "/srv/models"}}


# -- the unit -----------------------------------------------------------------------------


def test_unit_runs_the_installed_command_with_its_config() -> None:
    unit = unit_text(
        user="sameer",
        home=Path("/home/sameer"),
        executable=Path("/home/sameer/.local/share/uv/tools/huddle/bin/huddle"),
        config=Path("/home/sameer/.config/huddle/huddle.yaml"),
        groups=["render", "video"],
    )
    assert unit.startswith(UNIT_MARKER)
    assert "User=sameer\n" in unit
    assert "ExecStart=/home/sameer/.local/share/uv/tools/huddle/bin/huddle serve\n" in unit
    assert "Environment=HUDDLE_CONFIG=/home/sameer/.config/huddle/huddle.yaml\n" in unit
    # A clean stop exits 143 under SIGTERM; without this every stop is a failure.
    assert "SuccessExitStatus=143\n" in unit
    assert "KillMode=control-group\n" in unit
    assert "SupplementaryGroups=render video\n" in unit
    assert "WantedBy=multi-user.target" in unit


def test_unit_omits_groups_the_system_lacks() -> None:
    unit = unit_text(user="u", home=Path("/h"), executable=Path("/h/huddle"), config=Path("/c"))
    assert "SupplementaryGroups" not in unit


def test_install_service_replaces_checkout_units(tmp_path: Path) -> None:
    (tmp_path / "huddle-agent.service").write_text("[Service]\nExecStart=/home/m/huddle/.venv\n")
    (tmp_path / "huddle.service").write_text("[Service]\nExecStart=uv run huddle serve\n")
    runner = FakeRunner()
    install_service("UNIT", runner, lambda _l: None, unit_dir=tmp_path)

    commands = runner.commands()
    assert "systemctl disable --now huddle-agent.service" in commands
    assert f"rm -f {tmp_path / 'huddle-agent.service'}" in commands
    assert "systemctl stop huddle.service" in commands, "the old unit holds the ports"
    tee = [call for call in runner.calls if call[0][0] == "tee"]
    assert tee == [(["tee", str(tmp_path / "huddle.service")], True, "UNIT")]
    assert commands[-3:] == [
        "systemctl daemon-reload",
        "systemctl enable huddle.service",
        "systemctl restart huddle.service",
    ]
    assert all(sudo for argv, sudo, _ in runner.calls), "every command here needs root"


def test_reinstalling_our_own_unit_does_not_stop_it_first(tmp_path: Path) -> None:
    (tmp_path / "huddle.service").write_text(f"{UNIT_MARKER}. ...\n")
    runner = FakeRunner()
    install_service("UNIT", runner, lambda _l: None, unit_dir=tmp_path)
    assert "systemctl stop huddle.service" not in runner.commands()


# -- firewall -------------------------------------------------------------------------------


def test_ufw_rules_are_limited_to_the_subnet() -> None:
    rules = [" ".join(c) for c in firewall_commands("ufw", "192.168.0.0/24", [8000, 50052])]
    assert rules == [
        "ufw allow from 192.168.0.0/24 to any port 8000 proto tcp comment huddle",
        "ufw allow from 192.168.0.0/24 to any port 50052 proto tcp comment huddle",
    ]
    removals = firewall_commands("ufw", "192.168.0.0/24", [8000], remove=True)
    assert [" ".join(c) for c in removals] == [
        "ufw delete allow from 192.168.0.0/24 to any port 8000 proto tcp"
    ]


def test_firewalld_rules_reload_after_adding() -> None:
    commands = firewall_commands("firewalld", "10.0.0.0/16", [8000])
    assert commands[0][:2] == ["firewall-cmd", "--permanent"]
    assert 'source address="10.0.0.0/16"' in commands[0][2]
    assert 'port port="8000"' in commands[0][2]
    assert commands[-1] == ["firewall-cmd", "--reload"]


# -- the whole run ----------------------------------------------------------------------------


@pytest.fixture
def no_downloads(monkeypatch: pytest.MonkeyPatch, fake_llama_server: Path) -> None:
    """Stand in for the llama.cpp install: the fake binary is already 'installed'."""

    def installed(*_args: object, **_kw: object) -> Path:
        return fake_llama_server

    monkeypatch.setattr("huddle.setup.install_llamacpp", installed)
    monkeypatch.setattr("huddle.setup.ensure_libraries", lambda *_a, **_k: None)


@pytest.mark.usefixtures("no_downloads")
def test_setup_writes_config_and_opens_the_firewall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("huddle.setup.detect_firewall", lambda _runner: "ufw")
    monkeypatch.setattr("huddle.setup.local_subnet", lambda: "192.168.0.0/24")
    runner = FakeRunner()
    lines: list[str] = []

    run_setup(
        cluster="lab",
        models_dir=None,
        service=False,
        firewall=True,
        echo=lines.append,
        runner=runner,
    )

    config = paths.default_config_file().read_text()
    assert 'cluster: "lab"' in config
    assert paths.default_models_dir().is_dir(), "the models dir exists before any download"
    assert "ufw allow from 192.168.0.0/24 to any port 8000 proto tcp comment huddle" in (
        runner.commands()
    )
    record = Installed.load(paths.state_dir() / "setup.json")
    assert record.firewall == "ufw"
    assert (record.subnet, record.ports) == ("192.168.0.0/24", [8000, 50052])


@pytest.mark.usefixtures("no_downloads")
def test_setup_closes_rules_for_a_network_it_left(monkeypatch: pytest.MonkeyPatch) -> None:
    Installed(firewall="ufw", subnet="10.1.0.0/16", ports=[8000]).save(
        paths.state_dir() / "setup.json"
    )
    monkeypatch.setattr("huddle.setup.detect_firewall", lambda _runner: "ufw")
    monkeypatch.setattr("huddle.setup.local_subnet", lambda: "192.168.0.0/24")
    runner = FakeRunner()
    run_setup(
        cluster=None,
        models_dir=None,
        service=False,
        firewall=True,
        echo=lambda _l: None,
        runner=runner,
    )
    assert "ufw delete allow from 10.1.0.0/16 to any port 8000 proto tcp" in runner.commands()


@pytest.mark.usefixtures("no_downloads")
def test_setup_keeps_an_existing_config() -> None:
    config = paths.default_config_file()
    config.parent.mkdir(parents=True)
    config.write_text("discovery:\n  cluster: mine\n")
    run_setup(
        cluster="other",
        models_dir=None,
        service=False,
        firewall=False,
        echo=lambda _l: None,
        runner=FakeRunner(),
    )
    assert config.read_text() == "discovery:\n  cluster: mine\n"


def test_setup_refuses_to_run_as_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("huddle.setup.os.geteuid", lambda: 0)
    with pytest.raises(SetupError, match="not root"):
        run_setup(
            cluster=None,
            models_dir=None,
            service=False,
            firewall=False,
            echo=print,
            runner=FakeRunner(),
        )


def test_uninstall_undoes_what_setup_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    Installed(service=True, firewall="ufw", subnet="192.168.0.0/24", ports=[8000]).save(
        paths.state_dir() / "setup.json"
    )
    (paths.llamacpp_root() / "b10976").mkdir(parents=True)
    paths.default_models_dir().mkdir(parents=True)
    runner = FakeRunner()
    run_uninstall(purge=False, echo=lambda _l: None, runner=runner)

    assert "ufw delete allow from 192.168.0.0/24 to any port 8000 proto tcp" in runner.commands()
    assert not paths.llamacpp_root().exists()
    assert paths.default_models_dir().exists(), "models survive without --purge"
    assert not (paths.state_dir() / "setup.json").exists()
