#!/usr/bin/env bash
#
# Linux wrapper: build and run manylinux_2_28 container to produce an optimized LLVM/MLIR toolchain
#
# Description:
#   Builds a manylinux_2_28 container image (arch-aware), mounts the repo and output directories,
#   then runs the in-container build script to produce a PGO + ThinLTO optimized LLVM/MLIR toolchain.
#   Uses ccache (mounted from the workspace) and emits a .tar.zst archive in the output directory.
#
# Usage:
#   scripts/toolchain/linux/build.sh <ref> <install_prefix> [targets]
#     ref            Git ref or commit SHA (e.g., llvmorg-20.1.8 or 179d30f...)
#     install_prefix Absolute path on the host for the final install (also where archive is written)
#     targets        LLVM_TARGETS_TO_BUILD (default: "X86;AArch64")
#
# Environment (forwarded into container):
#   TOOLCHAIN_CLEAN=1           Wipe previous builds inside the container workspace
#   TOOLCHAIN_STAGE_FROM/TO     Limit stages (e.g., 2 and 2 for Stage2 only)
#   TOOLCHAIN_HOST_TRIPLE       Override computed host triple
#   TOOLCHAIN_CPU_FLAGS         Extra tuning flags (e.g., -march=haswell)
#
# Outputs:
#   - Installs into <install_prefix>
#   - Creates <install_prefix>/llvm-mlir_<ref>_linux_<arch>_<targets>_opt.tar.zst
#
# Example:
#   scripts/toolchain/linux/build.sh llvmorg-20.1.8 "$PWD/llvm-install" X86
#
set -euo pipefail

# Arguments: <ref-commit> <install_prefix> <targets>
REF=${1:?ref}
PREFIX=${2:?install_prefix}
TARGETS=${3:-"X86;AArch64"}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)

ARCH=$(uname -m)
BASE_IMAGE="quay.io/pypa/manylinux_2_28_x86_64"
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
  BASE_IMAGE="quay.io/pypa/manylinux_2_28_aarch64"
fi

IMG_TAG="llvm-mlir-manylinux-2_28:${ARCH}"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"

# Build container image
sudo docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -f "$DOCKERFILE" -t "$IMG_TAG" "$SCRIPT_DIR"

# Ensure output dir exists and ccache dir
mkdir -p "$PREFIX"
CCACHE_HOST_DIR="$ROOT_DIR/.ccache"
mkdir -p "$CCACHE_HOST_DIR"

# Determine path to in-container script once mounted at /work
# If ROOT_DIR is mounted at /work, then SCRIPT_DIR becomes /work${REL_DIR}
REL_DIR="${SCRIPT_DIR#${ROOT_DIR}}"
IN_CONTAINER_SCRIPT="/work${REL_DIR}/in-container.sh"

# Run build inside container (privileged for perf)
sudo docker run --rm --privileged \
  -u $(id -u):$(id -g) \
  -v "$ROOT_DIR":/work:rw \
  -v "$PREFIX":/out:rw \
  -v "$CCACHE_HOST_DIR":/work/.ccache:rw \
  -e REF="$REF" \
  -e TARGETS="$TARGETS" \
  -e PREFIX="/out" \
  -e TOOLCHAIN_CLEAN="${TOOLCHAIN_CLEAN:-0}" \
  -e TOOLCHAIN_STAGE_FROM="${TOOLCHAIN_STAGE_FROM:-0}" \
  -e TOOLCHAIN_STAGE_TO="${TOOLCHAIN_STAGE_TO:-2}" \
  -e TOOLCHAIN_HOST_TRIPLE="${TOOLCHAIN_HOST_TRIPLE:-}" \
  -e TOOLCHAIN_CPU_FLAGS="${TOOLCHAIN_CPU_FLAGS:-}" \
  -e CCACHE_DIR="/work/.ccache" \
  "$IMG_TAG" \
  bash "$IN_CONTAINER_SCRIPT"

echo "Linux build completed at $PREFIX"
