#!/usr/bin/env bash
#
# macOS LLVM/MLIR optimized toolchain builder (PGO + ThinLTO)
#
# Description:
#   Builds a multi-stage optimized LLVM/MLIR toolchain on macOS using:
#   - Stage0: clang + compiler-rt (profile runtime)
#   - Stage1: instrumented build + tests to gather profiles
#   - Stage2: PGO + ThinLTO final toolchain
#   Uses ccache as the compiler launcher, installs FileCheck (LLVM_INSTALL_UTILS),
#   targets macOS 11+ SDK, and emits a .tar.zst archive.
#
# Usage:
#   scripts/toolchain/macos/build.sh <ref> <install_prefix> [targets]
#     ref            Git ref or tag (e.g., llvmorg-19.1.7 or a commit SHA)
#     install_prefix Absolute install directory for the final toolchain
#     targets        LLVM_TARGETS_TO_BUILD (default: "X86;AArch64")
#
# Environment:
#   TOOLCHAIN_CLEAN=1           Wipe prior builds before building (default: 0)
#   TOOLCHAIN_STAGE_FROM/TO     Limit stages (e.g., 2 and 2 for Stage2 only)
#   TOOLCHAIN_HOST_TRIPLE       Override computed host triple
#   TOOLCHAIN_CPU_FLAGS         Extra C/C++ flags (e.g., -mcpu=native)
#   MACOSX_DEPLOYMENT_TARGET    Override SDK deployment target (default: 11.0)
#   CCACHE_*                    ccache knobs (COMPRESS, MAXSIZE, etc.)
#
# Outputs:
#   - Installs into <install_prefix>
#   - Creates archive: llvm-mlir_<ref>_macos_<arch>_<targets>_opt.tar.zst in CWD
#
# Examples:
#   TOOLCHAIN_STAGE_FROM=2 TOOLCHAIN_STAGE_TO=2 \
#     scripts/toolchain/macos/build.sh llvmorg-19.1.7 "$PWD/llvm-install" X86
#   TOOLCHAIN_CLEAN=1 \
#     scripts/toolchain/macos/build.sh 179d30f8 "$PWD/llvm-install" "X86;AArch64"
#
set -euo pipefail

# Arguments: <ref> <install_prefix> <targets>
REF=${1:?ref}
PREFIX=${2:?install_prefix}
TARGETS_ARG=${3:-}
# Normalize 'auto' to empty so we compute from host
if [[ "${TARGETS_ARG:-}" == "auto" ]]; then TARGETS_ARG=""; fi

WORKDIR=$(pwd)

# Incremental controls (set TOOLCHAIN_CLEAN=1 to wipe; set TOOLCHAIN_STAGE_FROM/TO to restrict stages)
CLEAN=${TOOLCHAIN_CLEAN:-0}
STAGE_FROM=${TOOLCHAIN_STAGE_FROM:-0}
STAGE_TO=${TOOLCHAIN_STAGE_TO:-2}

if [[ "$CLEAN" == "1" ]]; then
  rm -rf llvm-project build_stage0 build_stage1 build_stage2 stage0-install stage1-install pgoprof
fi
mkdir -p pgoprof/raw

# Determine host and macOS settings
UNAME_ARCH=$(uname -m)
MACOS_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-11.0}
export MACOSX_DEPLOYMENT_TARGET="$MACOS_DEPLOYMENT_TARGET"
# Prefer stable host triple without OS minor version to keep artifacts portable
if [[ "$UNAME_ARCH" == "arm64" || "$UNAME_ARCH" == "aarch64" ]]; then
  HOST_TRIPLE_COMPUTED="arm64-apple-darwin"
  OSX_ARCHS="arm64"
  HOST_TARGET="AArch64"
else
  HOST_TRIPLE_COMPUTED="x86_64-apple-darwin"
  OSX_ARCHS="x86_64"
  HOST_TARGET="X86"
fi
# Compute default targets from host unless overridden
if [[ -n "${TARGETS_ARG}" ]]; then
  TARGETS="${TARGETS_ARG}"
else
  TARGETS="${HOST_TARGET}"
fi
HOST_TRIPLE=${TOOLCHAIN_HOST_TRIPLE:-$HOST_TRIPLE_COMPUTED}
# Resolve SDK if available (falls back to CMake default)
if command -v xcrun >/dev/null 2>&1; then
  OSX_SYSROOT=$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)
else
  OSX_SYSROOT=""
fi

# Clone or update llvm-project
if [[ -d llvm-project/.git ]]; then
  pushd llvm-project >/dev/null
  git remote set-url origin https://github.com/llvm/llvm-project.git || true
  if git fetch --depth 1 origin "$REF" 2>/dev/null; then
    git checkout -f FETCH_HEAD
  else
    git fetch --tags --depth 1 origin
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

# Install uv and lit
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv tool install lit
LIT_BIN=$(which lit)

# Optional ccache as compiler launcher
LAUNCHER_ARGS=()
if command -v ccache >/dev/null 2>&1; then
  LAUNCHER_ARGS+=( -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache )
  export CCACHE_COMPRESS=1
  export CCACHE_COMPRESSLEVEL=${CCACHE_COMPRESSLEVEL:-19}
  export CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime
  export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-10G}
  CCACHE_ON=1
else
  CCACHE_ON=0
fi

# CPU tuning: favor modern x86_64 when building on Intel; tune Apple Silicon on arm64
CPU_FLAGS_DEFAULT=""
if [[ "$UNAME_ARCH" == "x86_64" ]]; then
  CPU_FLAGS_DEFAULT="-march=haswell -mtune=haswell"
elif [[ "$UNAME_ARCH" == "arm64" || "$UNAME_ARCH" == "aarch64" ]]; then
  CPU_FLAGS_DEFAULT="-mcpu=apple-m1 -mtune=apple-m1"
fi
# Allow override via TOOLCHAIN_CPU_FLAGS
CPU_FLAGS=${TOOLCHAIN_CPU_FLAGS:-$CPU_FLAGS_DEFAULT}

# Share common CMake args across stages
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
  -DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}
  -DCMAKE_OSX_ARCHITECTURES=${OSX_ARCHS}
)
if [[ -n "${OSX_SYSROOT:-}" ]]; then
  COMMON_LLVM_ARGS+=( -DCMAKE_OSX_SYSROOT="${OSX_SYSROOT}" )
fi
# Append launcher and ccache build knob if present
COMMON_LLVM_ARGS+=( "${LAUNCHER_ARGS[@]}" )
if (( CCACHE_ON == 1 )); then
  COMMON_LLVM_ARGS+=( -DLLVM_CCACHE_BUILD=ON )
fi

# Stage0: build clang + compiler-rt(profile) to drive subsequent builds
if (( STAGE_FROM <= 0 && 0 <= STAGE_TO )); then
  cmake -S llvm-project/llvm -B build_stage0 \
    "${COMMON_LLVM_ARGS[@]}" \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_BUILD_TESTS=OFF \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
    -DCOMPILER_RT_BUILD_PROFILE=ON \
    -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
    -DCOMPILER_RT_BUILD_XRAY=OFF \
    -DCOMPILER_RT_BUILD_MEMPROF=OFF \
    -DCMAKE_INSTALL_PREFIX="$WORKDIR/stage0-install"
  cmake --build build_stage0 --target install --config Release
fi

# Prefer stage0 clang if present; otherwise fall back to system clang
if [[ -x "$WORKDIR/stage0-install/bin/clang" && -x "$WORKDIR/stage0-install/bin/clang++" ]]; then
  export CC="$WORKDIR/stage0-install/bin/clang"
  export CXX="$WORKDIR/stage0-install/bin/clang++"
else
  export CC=${CC:-clang}
  export CXX=${CXX:-clang++}
fi

# Stage1: instrumented build to generate profiles during build/tests
if (( STAGE_FROM <= 1 && 1 <= STAGE_TO )); then
  export LLVM_PROFILE_FILE="$WORKDIR/pgoprof/raw/%p-%m.profraw"
  INSTR_FLAGS="-fprofile-instr-generate ${CPU_FLAGS}"
  cmake -S llvm-project/llvm -B build_stage1 \
    "${COMMON_LLVM_ARGS[@]}" \
    -DLLVM_INCLUDE_TESTS=ON -DLLVM_BUILD_TESTS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DCMAKE_C_COMPILER="${CC}" -DCMAKE_CXX_COMPILER="${CXX}" -DCMAKE_ASM_COMPILER="${CC}" \
    -DCMAKE_C_FLAGS="${INSTR_FLAGS}" -DCMAKE_CXX_FLAGS="${INSTR_FLAGS}" \
    -DLLVM_EXTERNAL_LIT="${LIT_BIN}" \
    -DCMAKE_INSTALL_PREFIX="$WORKDIR/stage1-install"
  cmake --build build_stage1 --target install --config Release
  # Run tests to produce .profraw
  cmake --build build_stage1 --config Release --target check-mlir || true
fi

# Merge profiles
RAW_DIR="$WORKDIR/pgoprof/raw"
PROFDATA="$WORKDIR/pgoprof/merged.profdata"
if compgen -G "$RAW_DIR/*.profraw" >/dev/null; then
  "$WORKDIR/stage0-install/bin/llvm-profdata" merge -output="$PROFDATA" "$RAW_DIR"/*.profraw || true
else
  echo "Warning: no .profraw collected; proceeding with empty profile" >&2
  : > "$PROFDATA"
fi

# Stage2: final PGO+ThinLTO build
if (( STAGE_FROM <= 2 && 2 <= STAGE_TO )); then
  USE_FLAGS="-fprofile-use=$PROFDATA -Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date ${CPU_FLAGS}"
  LD_FLAGS="-Wl,-no_warn_duplicate_libraries"
  cmake -S llvm-project/llvm -B build_stage2 \
    "${COMMON_LLVM_ARGS[@]}" \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DCMAKE_C_COMPILER="${CC}" -DCMAKE_CXX_COMPILER="${CXX}" -DCMAKE_ASM_COMPILER="${CC}" \
    -DCMAKE_C_FLAGS="${USE_FLAGS}" -DCMAKE_CXX_FLAGS="${USE_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${LD_FLAGS}" -DCMAKE_SHARED_LINKER_FLAGS="${LD_FLAGS}" \
    -DLLVM_EXTERNAL_LIT="${LIT_BIN}" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}"
  cmake --build build_stage2 --target install --config Release
fi

# Prune any non-essential tools from install (clang/bolt/lld should not be present)
if [[ -d "$PREFIX/bin" ]]; then
  rm -f "$PREFIX/bin"/clang* "$PREFIX/bin"/clang-?* "$PREFIX/bin"/clang++* \
        "$PREFIX/bin"/clangd "$PREFIX/bin"/clang-format* "$PREFIX/bin"/clang-tidy* \
        "$PREFIX/bin"/lld* "$PREFIX/bin"/llvm-bolt "$PREFIX/bin"/perf2bolt 2>/dev/null || true
fi
rm -rf "$PREFIX/lib/clang" 2>/dev/null || true

# Strip binaries
if command -v strip >/dev/null 2>&1; then
  find "$PREFIX/bin" -type f -perm -111 -exec strip -S {} + 2>/dev/null || true
  find "$PREFIX/lib" -name "*.a" -exec strip -S {} + 2>/dev/null || true
fi

# Emit compressed archive (.tar.zst)
if command -v gtar >/dev/null 2>&1; then TAR=gtar; else TAR=tar; fi
ART_DIR="$WORKDIR"
SAFE_TARGETS=${TARGETS//;/_}
ARCHIVE_NAME="llvm-mlir_${REF}_macos_${UNAME_ARCH}_${SAFE_TARGETS}_opt.tar.zst"
if command -v zstd >/dev/null 2>&1; then
  ( cd "${PREFIX}" && $TAR -cf - . | zstd -T0 -19 -o "${ART_DIR}/${ARCHIVE_NAME}" ) || true
else
  ( cd "${PREFIX}" && $TAR --zstd -cf "${ART_DIR}/${ARCHIVE_NAME}" . ) || true
fi

# Report
echo "macOS build completed at $PREFIX (incremental, PGO+ThinLTO, Zstd, $([[ $CCACHE_ON -eq 1 ]] && echo ccache || echo no-ccache), HOST_TRIPLE=${HOST_TRIPLE}, TARGETS=${TARGETS})"
echo "Archive: ${ART_DIR}/${ARCHIVE_NAME}"
