#!/usr/bin/env bash
# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

#
# Linux (manylinux_2_28) LLVM/MLIR optimized toolchain builder (PGO + ThinLTO + BOLT)
#
# Description:
#   Runs inside a manylinux_2_28 container to build a multi-stage optimized LLVM/MLIR toolchain:
#   - Stage0: clang + compiler-rt (profile runtime)
#   - Stage1: instrumented build + tests to gather profiles
#   - Stage2: PGO + ThinLTO final toolchain; runs BOLT post-link optimization on hot executables
#   Uses ccache, installs FileCheck (LLVM_INSTALL_UTILS), and emits a .tar.zst archive in /out.
#
# Usage (inside container; invoked by scripts/toolchain/linux/build.sh):
#   scripts/toolchain/linux/in-container.sh
#   Required env: REF, PREFIX
#     REF      Git ref/SHA (e.g., llvmorg-19.1.7 or 179d30f...)
#     PREFIX   Absolute path where the final toolchain will be installed
#
# Environment:
#   TARGETS                    LLVM_TARGETS_TO_BUILD (default: "X86;AArch64")
#   TOOLCHAIN_CLEAN=1          Wipe prior builds before building (default: 0)
#   TOOLCHAIN_STAGE_FROM/TO    Limit stages (e.g., 2 and 2 for Stage2 only)
#   TOOLCHAIN_HOST_TRIPLE      Override computed host triple
#   TOOLCHAIN_CPU_FLAGS        Extra CPU tuning (e.g., -march=haswell)
#   CCACHE_DIR                 Cache directory mounted from host (default: /work/.ccache)
#   CCACHE_*                   ccache knobs (COMPRESS, MAXSIZE, etc.)
#
# Outputs:
#   - Installs into $PREFIX
#   - Creates archive: /out/llvm-mlir_<ref>_linux_<arch>_<targets>_opt.tar.zst
#
# Examples (host invokes wrapper):
#   scripts/toolchain/linux/build.sh llvmorg-20.1.8 $PWD/llvm-install X86
#   TOOLCHAIN_STAGE_FROM=2 TOOLCHAIN_STAGE_TO=2 scripts/toolchain/linux/build.sh <sha> $PWD/llvm-install AArch64
#
set -euo pipefail

: "${REF:?REF (commit) not set}"
: "${PREFIX:?PREFIX not set}"
TARGETS_ENV=${TARGETS:-}
# Normalize 'auto' to empty so we compute from host
if [[ "${TARGETS_ENV:-}" == "auto" ]]; then TARGETS_ENV=""; fi

export DEBIAN_FRONTEND=noninteractive

cd /work
CLEAN=${TOOLCHAIN_CLEAN:-0}
STAGE_FROM=${TOOLCHAIN_STAGE_FROM:-0}
STAGE_TO=${TOOLCHAIN_STAGE_TO:-2}
if [[ "$CLEAN" == "1" ]]; then
  rm -rf llvm-project build_stage0 build_stage1 build_stage2 /tmp/pgoprof stage0-install stage1-install
fi
mkdir -p /tmp/pgoprof

# Clone/update llvm-project at specific commit
if [[ -d llvm-project/.git ]]; then
  pushd llvm-project >/dev/null
  git remote set-url origin https://github.com/llvm/llvm-project.git || true
  git fetch --depth 1 origin ${REF}
  git checkout -f FETCH_HEAD
  popd >/dev/null
else
  git clone --depth 1 https://github.com/llvm/llvm-project.git
  cd llvm-project
  git fetch origin ${REF} --depth 1
  git checkout ${REF}
  cd ..
fi

# Install uv and lit (no venv)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv tool install lit
LIT_BIN=$(which lit)

# Optional ccache
LAUNCHER_ARGS=()
if command -v ccache >/dev/null 2>&1; then
  export CCACHE_DIR=${CCACHE_DIR:-/work/.ccache}
  mkdir -p "$CCACHE_DIR" || true
  export CCACHE_COMPRESS=1
  export CCACHE_COMPRESSLEVEL=${CCACHE_COMPRESSLEVEL:-19}
  export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-10G}
  LAUNCHER_ARGS+=( -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache )
  CCACHE_ON=1
else
  CCACHE_ON=0
fi

# Host triple, CPU tuning, default TARGETS from host
UNAME_ARCH=$(uname -m)
if [[ "$UNAME_ARCH" == "x86_64" ]]; then
  CPU_FLAGS=${TOOLCHAIN_CPU_FLAGS:-"-march=x86-64-v2 -mtune=haswell"}
  HOST_TRIPLE_COMPUTED="x86_64-unknown-linux-gnu"
  HOST_TARGET="X86"
elif [[ "$UNAME_ARCH" == "aarch64" || "$UNAME_ARCH" == "arm64" ]]; then
  CPU_FLAGS=${TOOLCHAIN_CPU_FLAGS:-""}
  HOST_TRIPLE_COMPUTED="aarch64-unknown-linux-gnu"
  HOST_TARGET="AArch64"
else
  CPU_FLAGS=${TOOLCHAIN_CPU_FLAGS:-""}
  HOST_TRIPLE_COMPUTED="${UNAME_ARCH}-unknown-linux-gnu"
  HOST_TARGET="X86"
fi
  HOST_TRIPLE=${TOOLCHAIN_HOST_TRIPLE:-$HOST_TRIPLE_COMPUTED}
  if [[ -n "$TARGETS_ENV" ]]; then
  TARGETS="$TARGETS_ENV"
else
  TARGETS="$HOST_TARGET"
fi

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
if (( CCACHE_ON == 1 )); then
  COMMON_LLVM_ARGS+=( -DLLVM_CCACHE_BUILD=ON )
fi

# Stage0: clang toolchain (+compiler-rt profile) for subsequent builds
if (( STAGE_FROM <= 0 && 0 <= STAGE_TO )); then
  cmake -S llvm-project/llvm -B build_stage0 \
    "${COMMON_LLVM_ARGS[@]}" \
    -DLLVM_ENABLE_PROJECTS="clang;bolt" \
    -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
    -DCOMPILER_RT_BUILD_PROFILE=ON \
    -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
    -DCOMPILER_RT_BUILD_XRAY=OFF \
    -DCOMPILER_RT_BUILD_MEMPROF=OFF \
    -DCMAKE_INSTALL_PREFIX=/work/stage0-install
  cmake --build build_stage0 --target install --config Release
fi
# Prefer stage0 clang if present; otherwise fall back to system clang
if [[ -x /work/stage0-install/bin/clang && -x /work/stage0-install/bin/clang++ ]]; then
  export CC=/work/stage0-install/bin/clang
  export CXX=/work/stage0-install/bin/clang++
else
  export CC=${CC:-clang}
  export CXX=${CXX:-clang++}
fi

# Stage1: instrumented build with tests, collect .profraw via check-mlir
PGO_DIR=/tmp/pgoprof
RAW_DIR=$PGO_DIR/raw
mkdir -p "$RAW_DIR"
if (( STAGE_FROM <= 1 && 1 <= STAGE_TO )); then
  export LLVM_PROFILE_FILE="$RAW_DIR/%p-%m.profraw"
  INSTR_FLAGS="-fprofile-instr-generate ${CPU_FLAGS}"
  RELOCS_FLAGS="-Wl,--emit-relocs"

  cmake -S llvm-project/llvm -B build_stage1 \
    "${COMMON_LLVM_ARGS[@]}" \
    -DLLVM_INCLUDE_TESTS=ON -DLLVM_BUILD_TESTS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_ASM_COMPILER=${CC} \
    -DCMAKE_C_FLAGS="$INSTR_FLAGS" -DCMAKE_CXX_FLAGS="$INSTR_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="$RELOCS_FLAGS" -DCMAKE_SHARED_LINKER_FLAGS="$RELOCS_FLAGS" \
    -DLLVM_EXTERNAL_LIT="$LIT_BIN" \
    -DCMAKE_INSTALL_PREFIX=/work/stage1-install

  cmake --build build_stage1 --config Release --target install
  # Run MLIR tests to generate .profraw
  cmake --build build_stage1 --config Release --target check-mlir || true
fi

# Merge to .profdata
PROFDATA=$PGO_DIR/merged.profdata
RAW_GLOB=$(shopt -s nullglob; echo $RAW_DIR/*.profraw)
if [[ -z "${RAW_GLOB// }" ]]; then
  echo "Warning: no .profraw collected; creating empty profdata" >&2
  : > "$PROFDATA"
else
  /work/stage0-install/bin/llvm-profdata merge -output="$PROFDATA" $RAW_DIR/*.profraw || true
fi

# Stage2: final PGO+ThinLTO build (use BOLT tools from stage0, do not install them)
if (( STAGE_FROM <= 2 && 2 <= STAGE_TO )); then
  USE_FLAGS="-fprofile-use=$PROFDATA -Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date ${CPU_FLAGS}"
  RELOCS_FLAGS="-Wl,--emit-relocs"
  cmake -S llvm-project/llvm -B build_stage2 \
    "${COMMON_LLVM_ARGS[@]}" \
    -DLLVM_INCLUDE_TESTS=ON -DLLVM_BUILD_TESTS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_ASM_COMPILER=${CC} \
    -DCMAKE_C_FLAGS="$USE_FLAGS" -DCMAKE_CXX_FLAGS="$USE_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="$RELOCS_FLAGS" -DCMAKE_SHARED_LINKER_FLAGS="$RELOCS_FLAGS" \
    -DLLVM_EXTERNAL_LIT="$LIT_BIN" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}"

  cmake --build build_stage2 --config Release

  # Perf-based BOLT profiles from stage2 tests
  PERF_DATA=/tmp/perf.data
  if command -v perf >/dev/null 2>&1; then
    perf record -e cycles:u -g -- \
      cmake --build build_stage2 --config Release --target check-mlir || true
  fi

  BIN_DIR=/work/build_stage2/bin
  PERF2BOLT="/work/stage0-install/bin/perf2bolt"
  LLVMBOLT="/work/stage0-install/bin/llvm-bolt"
  if [[ -x "$PERF2BOLT" && -x "$LLVMBOLT" && -f "$PERF_DATA" ]]; then
    for exe in "$BIN_DIR/mlir-opt" "$BIN_DIR/mlir-translate"; do
      if [[ -x "$exe" ]]; then
        FDATA="$exe.fdata"
        "$PERF2BOLT" -p "$PERF_DATA" -o "$FDATA" "$exe" || true
        if [[ -s "$FDATA" ]]; then
          "$LLVMBOLT" "$exe" -o "$exe.bolt" -data "$FDATA" \
            -reorder-blocks=cache -reorder-functions=hfsort \
            -split-functions -split-all-cold -icf=1 -use-gnu-eh-frame || true
          if [[ -s "$exe.bolt" ]]; then mv -f "$exe.bolt" "$exe"; fi
        fi
      fi
    done
  fi

  # Final install after BOLT
  cmake --build build_stage2 --config Release --target install
fi

# Prune any non-essential tools that may have slipped into the install (keep MLIR/LLVM libs & tools only)
if [[ -d "$PREFIX/bin" ]]; then
  rm -f "$PREFIX/bin"/clang* "$PREFIX/bin"/clang-?* "$PREFIX/bin"/clang++* \
        "$PREFIX/bin"/clangd "$PREFIX/bin"/clang-format* "$PREFIX/bin"/clang-tidy* \
        "$PREFIX/bin"/lld* "$PREFIX/bin"/llvm-bolt "$PREFIX/bin"/perf2bolt 2>/dev/null || true
fi
rm -rf "$PREFIX/lib/clang" 2>/dev/null || true

# Strip binaries and libs
if command -v strip >/dev/null 2>&1; then
  find "$PREFIX/bin" -type f -perm -111 -exec strip -s {} + 2>/dev/null || true
  find "$PREFIX/lib" \( -name "*.a" -o -name "*.so*" \) -exec strip -g {} + 2>/dev/null || true
fi

# Emit compressed archive (.tar.zst) into /out (max compression)
SAFE_TARGETS=${TARGETS//;/_}
ARCHIVE_NAME="llvm-mlir_${REF}_linux_${UNAME_ARCH}_${SAFE_TARGETS}_opt.tar.zst"
( cd "${PREFIX}" && tar -I 'zstd -T0 -19' -cf "/out/${ARCHIVE_NAME}" . ) || true

echo "Install completed at ${PREFIX} (incremental, PGO+ThinLTO, $( [[ $CCACHE_ON -eq 1 ]] && echo cache=ccache || echo no-cache ), Zstd, HOST_TRIPLE=${HOST_TRIPLE}, TARGETS=${TARGETS})"
echo "Archive: /out/${ARCHIVE_NAME}"
