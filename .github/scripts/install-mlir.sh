#!/bin/bash

set -euo pipefail

# Detect platform and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
  linux) PLATFORM="linux" ;;
  darwin) PLATFORM="macos" ;;
  *) echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

case "$ARCH" in
  x86_64) ARCH_SUFFIX="x86_64" ;;
  arm64|aarch64) ARCH_SUFFIX="arm64" ;;
  *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

# Set asset name and URL
TAG="test-release"

if [[ "$PLATFORM" == "linux" && "$ARCH_SUFFIX" == "x86_64" ]]; then
  ASSET_NAME="ubuntu-24.04-archive.zip"
elif [[ "$PLATFORM" == "linux" && "$ARCH_SUFFIX" == "arm64" ]]; then
  ASSET_NAME="ubuntu-24.04-arm-archive.zip"
elif [[ "$PLATFORM" == "macos" && "$ARCH_SUFFIX" == "x86_64" ]]; then
  ASSET_NAME="macos-15-intel-archive.zip"
elif [[ "$PLATFORM" == "macos" && "$ARCH_SUFFIX" == "arm64" ]]; then
  ASSET_NAME="macos-15-archive.zip"
else
  echo "Unsupported platform/architecture combination: ${PLATFORM}/${ARCH_SUFFIX}" >&2
  exit 1
fi

RELEASE_URL="https://github.com/burgholzer/portable-mlir-toolchain/releases/download/${TAG}/${ASSET_NAME}"

# Download asset
echo "Downloading $ASSET_NAME from $RELEASE_URL..."
curl -L -o "$ASSET_NAME" "$RELEASE_URL"

# Unzip asset
echo "Unzipping $ASSET_NAME..."
unzip -q "$ASSET_NAME"

# Find archive after unzip
ARCHIVE_PATH=$(find . -name "*.tar.zst" -print -quit)
if [[ -z "$ARCHIVE_PATH" ]]; then
  echo "No archive found after unzip of $ASSET_NAME." >&2
  exit 1
fi

# Check for zstd
if ! command -v zstd >/dev/null 2>&1; then
  echo "zstd not found. Please install zstd (brew install zstd or sudo apt-get install zstd)." >&2
  exit 1
fi

# Unpack archive
echo "Extracting $ARCHIVE_PATH..."
zstd -d "$ARCHIVE_PATH" --output-dir-flat .
tar -xf "${ARCHIVE_PATH%.zst}"

echo "Done. Archive extracted."
