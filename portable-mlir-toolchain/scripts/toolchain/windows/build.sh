#!/usr/bin/env bash
#
# Windows LLVM/MLIR optimized toolchain builder (PGO + ThinLTO)
#
# Description:
#   Builds a multi-stage optimized LLVM/MLIR toolchain on Windows using GitHub Actions' bash
#   (Git Bash) environment. Mirrors the PowerShell implementation but requires no pwsh.
#   Stages:
#     - Stage0: clang + compiler-rt (profile runtime)
#     - Stage1: instrumented build + lit tests to collect profiles
#     - Stage2: final PGO + ThinLTO toolchain
#   Uses ccache when available and emits a .tar.zst archive via tar | zstd.
#
# Usage:
#   scripts/toolchain/windows/build.sh <ref> <install_prefix> [targets]
#     ref            Git ref/tag/commit (e.g., llvmorg-20.1.8 or SHA)
#     install_prefix Absolute install directory for the final toolchain
#     targets        LLVM_TARGETS_TO_BUILD (default: auto → host)
#
# Environment:
#   TOOLCHAIN_CLEAN=1           Clean before building
#   TOOLCHAIN_STAGE_FROM/TO     Limit stages (e.g., 2 and 2 for Stage2 only)
#   TOOLCHAIN_HOST_TRIPLE       Override computed host triple
#   TOOLCHAIN_CPU_FLAGS         Extra flags (e.g., -march=haswell)
#
# Outputs:
#   - Installs into <install_prefix>
#   - Creates llvm-mlir_<ref>_windows_<arch>_<targets>_opt.tar.zst in the current working directory
#
set -euo pipefail

REF=${1:?ref}
PREFIX=${2:?install_prefix}
TARGETS_ARG=${3:-auto}
# Normalize 'auto' to empty so we compute from host
if [[ "${TARGETS_ARG:-}" == "auto" ]]; then TARGETS_ARG=""; fi

WORKDIR=$(pwd)
CLEAN=${TOOLCHAIN_CLEAN:-0}
STAGE_FROM=${TOOLCHAIN_STAGE_FROM:-0}
STAGE_TO=${TOOLCHAIN_STAGE_TO:-2}

if [[ "$CLEAN" == "1" ]]; then
  rm -rf llvm-project build_stage0 build_stage1 build_stage2 stage0-install stage1-install pgoprof || true
fi
mkdir -p pgoprof/raw

# Host triple/arch and default targets
UNAME_ARCH=$(uname -m)
case "$UNAME_ARCH" in
  x86_64|amd64) HOST_TRIPLE_COMPUTED="x86_64-pc-windows-msvc"; HOST_TARGET="X86" ;;
  aarch64|arm64) HOST_TRIPLE_COMPUTED="aarch64-pc-windows-msvc"; HOST_TARGET="AArch64" ;;
  *) HOST_TRIPLE_COMPUTED="x86_64-pc-windows-msvc"; HOST_TARGET="X86" ;;
esac
if [[ -n "${TARGETS_ARG}" ]]; then TARGETS="${TARGETS_ARG}"; else TARGETS="${HOST_TARGET}"; fi
HOST_TRIPLE=${TOOLCHAIN_HOST_TRIPLE:-$HOST_TRIPLE_COMPUTED}

# CPU tuning
if [[ "$UNAME_ARCH" == "x86_64" || "$UNAME_ARCH" == "amd64" ]]; then
  CPU_FLAGS_DEFAULT="-march=haswell -mtune=haswell"
else
  CPU_FLAGS_DEFAULT=""
fi
CPU_FLAGS=${TOOLCHAIN_CPU_FLAGS:-$CPU_FLAGS_DEFAULT}

# Clone or update llvm-project
if [[ -d llvm-project/.git ]]; then
  pushd llvm-project >/dev/null
  git remote set-url origin https://github.com/llvm/llvm-project.git || true
  if git fetch --depth 1 origin "$REF" 2>/dev/null; then
    git checkout -f FETCH_HEAD
  else
    git fetch --tags --depth 1 origin || true
    git checkout -f "$REF" || git checkout -f "origin/$REF"
  fi
  popd >/dev/null
else
  if [[ "$REF" =~ ^llvmorg- ]]; then
    git clone --depth 1 --branch "$REF" https://github.com/llvm/llvm-project.git
  else
    git clone --depth 1 https://github.com/llvm/llvm-project.git
    pushd llvm-project >/dev/null
    git fetch origin "$REF" --depth 1
    git checkout -f FETCH_HEAD
    popd >/dev/null
  fi
fi

# Install uv and lit (no venv)
if ! command -v uv >/dev/null 2>&1; then
  # uv Windows installer is a PowerShell script, but GitHub Windows runners also ship a sh installer
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv tool install lit
LIT_BIN=$(which lit)

# Optional ccache
LAUNCHER_ARGS=()
if command -v ccache >/dev/null 2>&1; then
  LAUNCHER_ARGS+=( -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DLLVM_CCACHE_BUILD=ON )
  CCACHE_ON=1
else
  CCACHE_ON=0
fi

# Helper to configure and build
cmake_gen() { cmake -S "$1" -B "$2" "${@:3}"; }
cmake_build() { cmake --build "$1" --config Release --target "${2:-install}"; }

# Common CMake args
COMMON_LLVM_ARGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_BUILD_TESTS=OFF
  -DLLVM_INCLUDE_EXAMPLES=OFF
  -DLLVM_ENABLE_ASSERTIONS=OFF
  -DLLVM_TARGETS_TO_BUILD="${TARGETS}"
  -DLLVM_ENABLE_LTO=Thin
  -DLLVM_ENABLE_ZSTD=ON
  -DLLVM_INSTALL_UTILS=ON
  -DLLVM_ENABLE_BINDINGS=OFF
  -DLLVM_HOST_TRIPLE=${HOST_TRIPLE}
  "${LAUNCHER_ARGS[@]}"
)

# Stage0: clang + compiler-rt (profile)
if (( STAGE_FROM <= 0 && 0 <= STAGE_TO )); then
  cmake_gen llvm-project/llvm build_stage0 \
    "${COMMON_LLVM_ARGS[@]}" \
    -DLLVM_ENABLE_PROJECTS=clang \
    -DLLVM_ENABLE_RUNTIMES=compiler-rt \
    -DCOMPILER_RT_BUILD_PROFILE=ON \
    -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
    -DCOMPILER_RT_BUILD_XRAY=OFF \
    -DCOMPILER_RT_BUILD_MEMPROF=OFF \
    -DCMAKE_INSTALL_PREFIX="$WORKDIR/stage0-install"
  cmake_build build_stage0 install
fi

# Prefer stage0 clang if present; otherwise system clang
if [[ -x "$WORKDIR/stage0-install/bin/clang.exe" && -x "$WORKDIR/stage0-install/bin/clang++.exe" ]]; then
  export CC="$WORKDIR/stage0-install/bin/clang.exe"
  export CXX="$WORKDIR/stage0-install/bin/clang++.exe"
else
  export CC=${CC:-clang}
  export CXX=${CXX:-clang++}
fi

# Stage1: instrumented build + tests to collect .profraw
RAW_DIR="$WORKDIR/pgoprof/raw"
if (( STAGE_FROM <= 1 && 1 <= STAGE_TO )); then
  export LLVM_PROFILE_FILE="$RAW_DIR/%p-%m.profraw"
  INSTR_FLAGS="-fprofile-instr-generate ${CPU_FLAGS}"
  cmake_gen llvm-project/llvm build_stage1 \
    "${COMMON_LLVM_ARGS[@]}" \
    -DLLVM_INCLUDE_TESTS=ON -DLLVM_BUILD_TESTS=ON \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DCMAKE_C_COMPILER="${CC}" -DCMAKE_CXX_COMPILER="${CXX}" -DCMAKE_ASM_COMPILER="${CC}" \
    -DCMAKE_C_FLAGS="${INSTR_FLAGS}" -DCMAKE_CXX_FLAGS="${INSTR_FLAGS}" \
    -DLLVM_EXTERNAL_LIT="${LIT_BIN}" \
    -DCMAKE_INSTALL_PREFIX="$WORKDIR/stage1-install"
  cmake_build build_stage1 install
  cmake_build build_stage1 check-mlir || true
fi

# Merge profiles to .profdata
PROFDATA="$WORKDIR/pgoprof/merged.profdata"
shopt -s nullglob
PROFRAW_FILES=("$RAW_DIR"/*.profraw)
if (( ${#PROFRAW_FILES[@]} == 0 )); then
  echo "Warning: no .profraw collected; proceeding with empty profile" >&2
  : > "$PROFDATA"
else
  "$WORKDIR/stage0-install/bin/llvm-profdata.exe" merge -output="$PROFDATA" "$RAW_DIR"/*.profraw || true
fi

# Stage2: final PGO+ThinLTO build
if (( STAGE_FROM <= 2 && 2 <= STAGE_TO )); then
  USE_FLAGS="-fprofile-use=$PROFDATA -Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date ${CPU_FLAGS}"
  cmake_gen llvm-project/llvm build_stage2 \
    "${COMMON_LLVM_ARGS[@]}" \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DCMAKE_C_COMPILER="${CC}" -DCMAKE_CXX_COMPILER="${CXX}" -DCMAKE_ASM_COMPILER="${CC}" \
    -DCMAKE_C_FLAGS="${USE_FLAGS}" -DCMAKE_CXX_FLAGS="${USE_FLAGS}" \
    -DLLVM_EXTERNAL_LIT="${LIT_BIN}" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}"
  cmake_build build_stage2 install
fi

# Prune non-essential tools (clang/bolt/lld)
if [[ -d "$PREFIX/bin" ]]; then
  rm -f "$PREFIX/bin"/clang* "$PREFIX/bin"/clang-?* "$PREFIX/bin"/clang++* \
        "$PREFIX/bin"/clangd* "$PREFIX/bin"/clang-format* "$PREFIX/bin"/clang-tidy* \
        "$PREFIX/bin"/lld* "$PREFIX/bin"/llvm-bolt* "$PREFIX/bin"/perf2bolt* 2>/dev/null || true
fi
rm -rf "$PREFIX/lib/clang" 2>/dev/null || true

# No stripping on Windows by default

# Emit compressed archive (.tar.zst)
if command -v gtar >/dev/null 2>&1; then TAR=gtar; else TAR=tar; fi
if ! command -v zstd >/dev/null 2>&1; then echo "zstd not found; please install it" >&2; exit 1; fi
SAFE_TARGETS=${TARGETS//;/_}
ARCHIVE_NAME="llvm-mlir_${REF}_windows_${UNAME_ARCH}_${SAFE_TARGETS}_opt.tar.zst"
(
  cd "${PREFIX}" && $TAR -cf - . | zstd -T0 -19 -o "${WORKDIR}/${ARCHIVE_NAME}"
) || true

echo "Windows build completed at ${PREFIX} (incremental, PGO+ThinLTO, Zstd, $( [[ $CCACHE_ON -eq 1 ]] && echo ccache || echo no-ccache ), HOST_TRIPLE=${HOST_TRIPLE}, TARGETS=${TARGETS})"
echo "Archive: ${WORKDIR}/${ARCHIVE_NAME}"
