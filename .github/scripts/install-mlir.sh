#!/bin/bash

# Usage: ./install-mlir.sh -t <tag> -p <installation directory>

set -euo pipefail

# Parse arguments
while getopts "t:p:*" opt; do
  case $opt in
    t) TAG="$OPTARG"
    ;;
    p) INSTALL_PREFIX="$OPTARG"
    ;;
    *) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done

if [ -z "$TAG" ]; then
  echo "Error: Tag (-t) is required"
  echo "Usage: $0 -t <tag> -p <installation directory>"
  exit 1
fi

if [ -z "${INSTALL_PREFIX:-}" ]; then
  echo "Error: Installation directory (-p) is required"
  echo "Usage: $0 -t <tag> -p <installation directory>"
  exit 1
fi

# Change to installation directory
pushd $INSTALL_PREFIX > /dev/null

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

# Determine download URL
RELEASE_URL="https://api.github.com/repos/burgholzer/portable-mlir-toolchain/releases/tags/${TAG}"
RELEASE_JSON=$(curl -L \
                    -H "Accept: application/vnd.github+json" \
                    -H "X-GitHub-Api-Version: 2022-11-28" \
                    "$RELEASE_URL")

ASSETS_URL=$(echo "$RELEASE_JSON" | jq -r '.assets_url')
ASSETS_JSON=$(curl -L \
                   -H "Accept: application/vnd.github+json" \
                   -H "X-GitHub-Api-Version: 2022-11-28" \
                   "$ASSETS_URL")

DOWNLOAD_URLS=$(echo "$ASSETS_JSON" | jq -r '.[].browser_download_url')

if [[ "$PLATFORM" == "linux" && "$ARCH_SUFFIX" == "x86_64" ]]; then
  DOWNLOAD_URL=$(echo "$DOWNLOAD_URLS" | grep '.*_linux_.*_X86.tar.zst')
elif [[ "$PLATFORM" == "linux" && "$ARCH_SUFFIX" == "arm64" ]]; then
  DOWNLOAD_URL=$(echo "$DOWNLOAD_URLS" | grep '.*_linux_.*_AArch64.tar.zst')
elif [[ "$PLATFORM" == "macos" && "$ARCH_SUFFIX" == "x86_64" ]]; then
  DOWNLOAD_URL=$(echo "$DOWNLOAD_URLS" | grep '.*_macos_.*_X86.tar.zst')
elif [[ "$PLATFORM" == "macos" && "$ARCH_SUFFIX" == "arm64" ]]; then
  DOWNLOAD_URL=$(echo "$DOWNLOAD_URLS" | grep '.*_macos_.*_AArch64.tar.zst')
else
  echo "Unsupported platform/architecture combination: ${PLATFORM}/${ARCH_SUFFIX}" >&2
  exit 1
fi

# Download asset
echo "Downloading asset from $DOWNLOAD_URL..."
curl -L -o "asset.tar.zst" "$DOWNLOAD_URL"

# Check for zstd
if ! command -v zstd >/dev/null 2>&1; then
  echo "zstd not found. Please install zstd (brew install zstd or sudo apt-get install zstd)." >&2
  exit 1
fi

# Unpack archive
echo "Extracting archive..."
zstd -d "asset.tar.zst" --output-dir-flat .
tar -xf "asset.tar"

# Return to original directory
popd > /dev/null

# Output instructions
echo "MLIR toolchain has been installed"
echo "Run the following commands to set up your environment:"
echo "  export LLVM_DIR=$PWD/lib/cmake/llvm"
echo "  export MLIR_DIR=$PWD/lib/cmake/mlir"
echo "  export PATH=$PWD/bin:\$PATH"
