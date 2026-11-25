#!/bin/sh
# agent-mem installer: downloads a prebuilt binary, verifies its checksum, installs it to
# ~/.agent-mem/bin and registers it with Claude Code, OpenCode, Codex CLI and Antigravity.
#
#   curl -fsSL https://raw.githubusercontent.com/hemanth090/agent-mem/main/install.sh | sh
#   curl -fsSL .../install.sh | sh -s -- --clients claude,opencode
#
# Environment:
#   AGENT_MEM_REPO       GitHub "owner/repo" to download from (default below)
#   AGENT_MEM_VERSION    release tag such as v0.1.0 (default: latest)
#   AGENT_MEM_BASE_URL   full base URL holding the archives (overrides repo/version; for mirrors/tests)
#   AGENT_MEM_DATA_DIR   where data and the installed binary live (default ~/.agent-mem)
# Any arguments are passed to `agent-mem install` (e.g. --dry-run, --clients codex, --no-model).

set -eu

REPO="${AGENT_MEM_REPO:-hemanth090/agent-mem}"
VERSION="${AGENT_MEM_VERSION:-latest}"
DATA_DIR="${AGENT_MEM_DATA_DIR:-$HOME/.agent-mem}"

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

detect_target() {
  os=$(uname -s)
  arch=$(uname -m)
  case "$os" in
    Darwin) os_part="apple-darwin" ;;
    Linux)  os_part="unknown-linux-gnu" ;;
    *) die "unsupported OS '$os' (supported: macOS, Linux). Build from source: cargo build --release" ;;
  esac
  case "$arch" in
    arm64|aarch64) arch_part="aarch64" ;;
    x86_64|amd64)  arch_part="x86_64" ;;
    *) die "unsupported CPU architecture '$arch'" ;;
  esac
  if [ "$os" = "Darwin" ] && [ "$arch_part" = "x86_64" ]; then
    die "Intel Macs are not supported by the prebuilt binaries (no ONNX Runtime build exists for x86_64 macOS). Build from source instead: cargo build --release && target/release/agent-mem install"
  fi
  printf '%s-%s' "$arch_part" "$os_part"
}

download() { # url dest
  if have curl; then
    curl -fsSL --retry 3 -o "$2" "$1"
  elif have wget; then
    wget -q -O "$2" "$1"
  else
    die "need curl or wget"
  fi
}

sha256_of() {
  if have shasum; then shasum -a 256 "$1" | cut -d' ' -f1
  elif have sha256sum; then sha256sum "$1" | cut -d' ' -f1
  else die "need shasum or sha256sum to verify the download"
  fi
}

TARGET=$(detect_target)
ARCHIVE="agent-mem-$TARGET.tar.gz"

if [ -n "${AGENT_MEM_BASE_URL:-}" ]; then
  BASE="$AGENT_MEM_BASE_URL"
elif [ "$VERSION" = "latest" ]; then
  BASE="https://github.com/$REPO/releases/latest/download"
else
  BASE="https://github.com/$REPO/releases/download/$VERSION"
fi

TMP=$(mktemp -d 2>/dev/null || mktemp -d -t agent-mem)
trap 'rm -rf "$TMP"' EXIT INT TERM

say "Downloading $ARCHIVE ..."
download "$BASE/$ARCHIVE" "$TMP/$ARCHIVE" || die "download failed: $BASE/$ARCHIVE (is there a release for $TARGET?)"
download "$BASE/$ARCHIVE.sha256" "$TMP/$ARCHIVE.sha256" || die "checksum download failed"

expected=$(cut -d' ' -f1 "$TMP/$ARCHIVE.sha256")
actual=$(sha256_of "$TMP/$ARCHIVE")
[ "$expected" = "$actual" ] || die "checksum mismatch (expected $expected, got $actual); refusing to install"

tar -xzf "$TMP/$ARCHIVE" -C "$TMP" || die "could not extract archive"
[ -x "$TMP/agent-mem" ] || die "archive did not contain an agent-mem binary"

# Registers the clients, copies the binary to $DATA_DIR/bin and pre-downloads the model.
"$TMP/agent-mem" --data-dir "$DATA_DIR" install "$@"

# Convenience: put `agent-mem` on PATH when ~/.local/bin exists or can be created.
INSTALLED="$DATA_DIR/bin/agent-mem"
if [ -x "$INSTALLED" ]; then
  LOCAL_BIN="$HOME/.local/bin"
  if mkdir -p "$LOCAL_BIN" 2>/dev/null && ln -sf "$INSTALLED" "$LOCAL_BIN/agent-mem" 2>/dev/null; then
    case ":$PATH:" in
      *":$LOCAL_BIN:"*) ;;
      *) say ""; say "Note: add $LOCAL_BIN to your PATH to use the \`agent-mem\` command (agents do not need this)." ;;
    esac
  fi
fi
