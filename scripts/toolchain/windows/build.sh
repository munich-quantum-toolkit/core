#!/usr/bin/env bash
#
# Windows LLVM/MLIR optimized toolchain builder (PGO + ThinLTO)
#
# Full bash implementation replacing PowerShell. See portable copy for details.
# This script mirrors portable-mlir-toolchain/scripts/toolchain/windows/build.sh.
#
set -euo pipefail

REF=${1:?ref}
PREFIX=${2:?install_prefix}
TARGETS_ARG=${3:-auto}
if [[ "${TARGETS_ARG:-}" == "auto" ]]; then TARGETS_ARG=""; fi

WORKDIR=$(pwd)
CLEAN=${TOOLCHAIN_CLEAN:-0}
STAGE_FROM=${TOOLCHAIN_STAGE_FROM:-0}
STAGE_TO=${TOOLCHAIN_STAGE_TO:-2}

if [[ "$CLEAN" == "1" ]]; then
  rm -rf llvm-project build_stage0 build_stage1 build_stage2 stage0-install stage1-install pgoprof || true
fi
mkdir -p pgoprof/raw

UNAME_ARCH=$(uname -m)
case "$UNAME_ARCH" in
  x86_64|amd64) HOST_TRIPLE_COMPUTED="x86_64-pc-windows-msvc"; HOST_TARGET="X86" ;;
  aarch64|arm64) HOST_TRIPLE_COMPUTED="aarch64-pc-windows-msvc"; HOST_TARGET="AArch64" ;;
  *) HOST_TRIPLE_COMPUTED="x86_64-pc-windows-msvc"; HOST_TARGET="X86" ;;
esac
if [[ -n "${TARGETS_ARG}" ]]; then TARGETS="${TARGETS_ARG}"; else TARGETS="${HOST_TARGET}"; fi
HOST_TRIPLE=${TOOLCHAIN_HOST_TRIPLE:-$HOST_TRIPLE_COMPUTED}

if [[ "$UNAME_ARCH" == "x86_64" || "$UNAME_ARCH" == "amd64" ]]; then
  CPU_FLAGS_DEFAULT="-march=haswell -mtune=haswell"
else
  CPU_FLAGS_DEFAULT=""
fi
CPU_FLAGS=${TOOLCHAIN_CPU_FLAGS:-$CPU_FLAGS_DEFAULT}

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

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv tool install lit
LIT_BIN=$(which lit)

LAUNCHER_ARGS=()
if command -v ccache >/dev/null 2>&1; then
  LAUNCHER_ARGS+=( -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DLLVM_CCACHE_BUILD=ON )
  CCACHE_ON=1
else
  CCACHE_ON=0
fi

cmake_gen() { cmake -S "$1" -B "$2" "${@:3}"; }
cmake_build() { cmake --build "$1" --config Release --target "${2:-install}"; }

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

if [[ -x "$WORKDIR/stage0-install/bin/clang.exe" && -x "$WORKDIR/stage0-install/bin/clang++.exe" ]]; then
  export CC="$WORKDIR/stage0-install/bin/clang.exe"
  export CXX="$WORKDIR/stage0-install/bin/clang++.exe"
else
  export CC=${CC:-clang}
  export CXX=${CXX:-clang++}
fi

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

PROFDATA="$WORKDIR/pgoprof/merged.profdata"
shopt -s nullglob
PROFRAW_FILES=("$RAW_DIR"/*.profraw)
if (( ${#PROFRAW_FILES[@]} == 0 )); then
  echo "Warning: no .profraw collected; proceeding with empty profile" >&2
  : > "$PROFDATA"
else
  "$WORKDIR/stage0-install/bin/llvm-profdata.exe" merge -output="$PROFDATA" "$RAW_DIR"/*.profraw || true
fi

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

if [[ -d "$PREFIX/bin" ]]; then
  rm -f "$PREFIX/bin"/clang* "$PREFIX/bin"/clang-?* "$PREFIX/bin"/clang++* \
        "$PREFIX/bin"/clangd* "$PREFIX/bin"/clang-format* "$PREFIX/bin"/clang-tidy* \
        "$PREFIX/bin"/lld* "$PREFIX/bin"/llvm-bolt* "$PREFIX/bin"/perf2bolt* 2>/dev/null || true
fi
rm -rf "$PREFIX/lib/clang" 2>/dev/null || true

if command -v gtar >/dev/null 2>&1; then TAR=gtar; else TAR=tar; fi
if ! command -v zstd >/dev/null 2>&1; then echo "zstd not found; please install it" >&2; exit 1; fi
SAFE_TARGETS=${TARGETS//;/_}
ARCHIVE_NAME="llvm-mlir_${REF}_windows_${UNAME_ARCH}_${SAFE_TARGETS}_opt.tar.zst"
(
  cd "${PREFIX}" && $TAR -cf - . | zstd -T0 -19 -o "${WORKDIR}/${ARCHIVE_NAME}"
) || true

echo "Windows build completed at ${PREFIX} (incremental, PGO+ThinLTO, Zstd, $( [[ $CCACHE_ON -eq 1 ]] && echo ccache || echo no-ccache ), HOST_TRIPLE=${HOST_TRIPLE}, TARGETS=${TARGETS})"
echo "Archive: ${WORKDIR}/${ARCHIVE_NAME}"
