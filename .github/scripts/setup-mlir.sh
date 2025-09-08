#!/usr/bin/env bash
# Cross-platform (Linux/macOS) setup for LLVM/MLIR using system package managers.
# Usage: bash .github/scripts/setup-mlir.sh <llvm_major_version>
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <llvm_major_version>" >&2
  exit 2
fi

LLVM_MAJOR="$1"
OS_NAME="$(uname -s)"

append_env() {
  # Append KEY=VALUE to GITHUB_ENV in a cross-platform safe way
  local kv="$1"
  if [[ -z "${GITHUB_ENV:-}" ]]; then
    echo "GITHUB_ENV not set. Are you running in GitHub Actions?" >&2
    exit 1
  fi
  echo "$kv" >> "$GITHUB_ENV"
}

append_path() {
  local p="$1"
  if [[ -z "${GITHUB_PATH:-}" ]]; then
    echo "GITHUB_PATH not set. Are you running in GitHub Actions?" >&2
    exit 1
  fi
  echo "$p" >> "$GITHUB_PATH"
}

download() {
  local url="$1" out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$out"
  else
    echo "Neither curl nor wget is available to download $url" >&2
    return 1
  fi
}

case "$OS_NAME" in
  Linux)
    echo "Installing LLVM/MLIR $LLVM_MAJOR on Linux via apt.llvm.org..."
    sudo apt-get update
    TMP_SH="${RUNNER_TEMP:-/tmp}/llvm_install.sh"
    download https://apt.llvm.org/llvm.sh "$TMP_SH"
    chmod +x "$TMP_SH"
    if sudo "$TMP_SH" "$LLVM_MAJOR"; then
      sudo apt-get install -y \
        libmlir-"$LLVM_MAJOR"-dev \
        llvm-"$LLVM_MAJOR"-tools \
        mlir-"$LLVM_MAJOR"-tools \
        clang-"$LLVM_MAJOR" \
        clang-tools-"$LLVM_MAJOR" || { echo "apt-get install failed"; exit 1; }
    else
      echo "Installation from apt.llvm.org script failed." >&2
      exit 1
    fi
    append_env "CC=clang-$LLVM_MAJOR"
    append_env "CXX=clang++-$LLVM_MAJOR"
    append_env "MLIR_DIR=/usr/lib/llvm-$LLVM_MAJOR/lib/cmake/mlir"
    append_env "LLVM_DIR=/usr/lib/llvm-$LLVM_MAJOR/lib/cmake/llvm"
    ;;
  Darwin)
    echo "Installing LLVM/MLIR $LLVM_MAJOR on macOS via Homebrew..."
    brew install "llvm@${LLVM_MAJOR}"
    LLVM_PREFIX="$(brew --prefix)/opt/llvm@${LLVM_MAJOR}"
    append_env "CC=${LLVM_PREFIX}/bin/clang"
    append_env "CXX=${LLVM_PREFIX}/bin/clang++"
    append_env "LDFLAGS=-L${LLVM_PREFIX}/lib/c++ -L${LLVM_PREFIX}/lib/unwind -lunwind"
    append_env "LLVM_DIR=${LLVM_PREFIX}/lib/cmake/llvm"
    append_env "MLIR_DIR=${LLVM_PREFIX}/lib/cmake/mlir"
    append_path "${LLVM_PREFIX}/bin"
    ;;
  *)
    echo "Unsupported OS for this script: $OS_NAME" >&2
    exit 1
    ;;

esac

echo "LLVM/MLIR setup complete for $OS_NAME."
