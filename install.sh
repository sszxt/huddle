#!/bin/sh
# Install Huddle on this machine:
#
#   curl -fsSL https://raw.githubusercontent.com/sszxt/huddle/main/install.sh | sh
#
# Run it on every Linux PC that should share its GPUs; they find each other on
# the local network. Re-run it to upgrade. Options after `sh -s --` are passed
# to `huddle setup`, e.g. `| sh -s -- --cluster lab`.
#
#   HUDDLE_REF=v0.2.0   install a tag or branch instead of main
#   HUDDLE_SOURCE=path  install from a local checkout (for testing a change)
set -eu

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] || die "Huddle runs on Linux only"
case "$(uname -m)" in
    x86_64 | amd64 | aarch64 | arm64) ;;
    *) die "no prebuilt llama.cpp for $(uname -m)" ;;
esac
[ "$(id -u)" -ne 0 ] || die "run this as your own user, not root: it asks for sudo when it needs to"
command -v curl >/dev/null 2>&1 || die "curl is required"

REF="${HUDDLE_REF:-main}"
SOURCE="${HUDDLE_SOURCE:-https://github.com/sszxt/huddle/archive/$REF.tar.gz}"

# uv brings its own Python, so the system's version does not matter.
if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
else
    say "installing uv (Python package manager) into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    UV="$HOME/.local/bin/uv"
    [ -x "$UV" ] || die "uv did not install; see https://docs.astral.sh/uv/"
fi

say "installing Huddle from $SOURCE"
"$UV" tool install --quiet --python 3.12 --force --reinstall "$SOURCE"
HUDDLE="$("$UV" tool dir --bin)/huddle"
[ -x "$HUDDLE" ] || die "huddle was not installed where uv said: $HUDDLE"

say "setting up this machine"
# stdin is this script when piped from curl; give setup (and sudo) the terminal.
if [ -r /dev/tty ] && [ -t 1 ]; then
    exec "$HUDDLE" setup "$@" </dev/tty
fi
exec "$HUDDLE" setup "$@"
