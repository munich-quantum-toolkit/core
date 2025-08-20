# Build LLVM+MLIR Optimized (PGO + ThinLTO + BOLT where applicable)

This composite action provisions a fast, optimized LLVM/MLIR toolchain for CI and local builds.
It will try to download a prebuilt bundle, and if not available, it builds from source and uploads the result for reuse.

Highlights

- Optimized builds: Profile-Guided Optimization (PGO) + ThinLTO everywhere; BOLT lightweight post-link tuning on Linux.
- Portable targets: LLVM_TARGETS_TO_BUILD is configurable; default is `auto` (host CPU family).
- Caching: integrates with ccache automatically when present (the action installs it on CI runners).
- Archives: .tar.zst on Linux, macOS, and Windows for better compression and faster transfers.
- Utilities: LLVM_INSTALL_UTILS=ON so FileCheck and tools are installed.

What it builds

- Stage0: clang + compiler-rt (profile runtime) to drive subsequent stages.
- Stage1: instrumented build to collect profiles via lit tests.
- Stage2: final optimized build with PGO profiles and ThinLTO; BOLT pass (Linux) on hot binaries where possible.

Platforms

- Linux: builds in a Docker manylinux_2_28 image, emits <key>.tar.zst.
- macOS: builds on the host, using macOS 11+ deployment target, emits <key>.tar.zst.
- Windows: builds on the host, emits <key>.tar.zst via tar | zstd.

Inputs

- version: Version selector (latest, a major like `20`, a full `20.1.8`, or a commit SHA). Required.
- targets: LLVM_TARGETS_TO_BUILD. Use `auto` (default) to build only the host family target (`X86` on x64, `AArch64` on ARM64).
- repo-for-assets: owner/repo to store release assets (default: current repo).
- release-tag: tag to use for assets (default: `llvm-binaries-opt`).
- publish: if true, upload newly built archives to the release (default: true).
- cache-download-only: if true, fail if asset missing instead of building (default: false).
- install-dir: install prefix for the toolchain (default: `${{ github.workspace }}/llvm-install`).

Outputs and environment

- outputs.install-dir: Installation prefix.
- outputs.llvm-dir: `$install_dir/lib/cmake/llvm`.
- outputs.mlir-dir: `$install_dir/lib/cmake/mlir`.
- Environment variables set for downstream steps:
  - LLVM_DIR, MLIR_DIR, and PATH updated to include `$install_dir/bin` (so `llvm-cov` and `FileCheck` are directly usable).

Asset naming

- Key: `llvm-mlir_<commit>_<OS>_<arch>_<targets>_opt` where `<targets>` reflects the resolved host target when `auto` is used.
- Archive: `<key>.tar.zst` on all platforms.

Usage: minimal

```yaml
- name: Setup LLVM+MLIR (optimized)
  id: llvm
  uses: ./.github/actions/build-llvm-mlir-opt
  with:
    version: 20.1.8
    targets: auto # or "X86;AArch64"

- name: Configure
  run: |
    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR="$LLVM_DIR" \
      -DMLIR_DIR="$MLIR_DIR"
```

Behavioral notes

- ccache usage: scripts auto-detect ccache; the action installs it on CI runners. When present, LLVM_CCACHE_BUILD=ON is enabled.
- macOS: uses MACOSX_DEPLOYMENT_TARGET=11.0 and an xcrun sysroot if available; host triple defaults to arm64-apple-darwin or x86_64-apple-darwin. Apple Silicon builds add `-mcpu=apple-m1 -mtune=apple-m1`.
- AArch64 tuning: on ARM, SSE/AVX don't exist; the scripts keep portable defaults. If you need more aggressive tuning, set `TOOLCHAIN_CPU_FLAGS` (e.g., `-mcpu=native` or `-march=armv8.2-a+crypto+fp16+dotprod`).

Advanced controls (passed through to scripts)

- TOOLCHAIN_CLEAN=1: wipe and rebuild from scratch.
- TOOLCHAIN_STAGE_FROM/TO: restrict to specific stages (e.g., `2..2` for final).
- TOOLCHAIN_HOST_TRIPLE: override computed host triple.
- TOOLCHAIN_CPU_FLAGS: override CPU tuning flags.
- MACOSX_DEPLOYMENT_TARGET: override macOS SDK target (default: 11.0).

Implementation details

- Linux builds in Docker (manylinux_2_28) with a ccache directory mounted from the workspace when available.
- Windows packaging uses `tar | zstd` to emit `.tar.zst`. The action extracts `.tar.zst` via 7zip on Windows.

FAQ

- Why `.tar.zst` on Windows? It compresses better and is faster to transfer than `.zip`, and the workflow ships 7zip to extract it.
- Where is FileCheck and llvm-cov? Installed as part of `LLVM_INSTALL_UTILS=ON` under `$install_dir/bin`; PATH is updated to include it.
