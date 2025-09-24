Portable LLVM/MLIR Optimized Toolchain (PGO + ThinLTO [+ BOLT on Linux])

Overview

- This directory is a self-contained, copy-ready bundle of scripts and CI to build optimized, portable LLVM/MLIR toolchains for Linux, macOS, and Windows.
- It is designed so you can copy the CONTENTS of this directory into a new repository and start experimenting immediately.

What's included

- scripts/toolchain/
  - linux/
    - build.sh: Host-side wrapper. Builds a manylinux_2_28 container image and runs the in-container build.
    - in-container.sh: The actual Linux build (multi-stage: Stage0/1/2) with PGO + ThinLTO, optional BOLT.
    - Dockerfile: Minimal manylinux_2_28 image with build tools.
  - macos/
    - build.sh: Host-side script performing Stage0/1/2 and producing an archive in the working directory.
  - windows/
    - build.sh: Full bash builder performing Stage0/1/2 and producing a .tar.zst archive.
    - build.ps1: Optional PowerShell variant (not used by the CI in this bundle).
- .github/actions/
  - build-llvm-mlir-opt/: Composite GitHub Action that can download prebuilt bundles or build locally using these scripts.
- .github/workflows/
  - build-portable-mlir-toolchain.yml: Example matrix workflow covering Linux, macOS (both Intel and Apple Silicon), and Windows (x64 and ARM). Uploads produced artifacts for inspection.

How to use in a new repository

1. Create a new empty repository.
2. Copy all files and folders INSIDE portable-mlir-toolchain/ into the repository root, keeping the structure (scripts/, .github/, etc.).
3. Commit and push. The workflow (workflow_dispatch) can then be manually triggered in GitHub Actions.

Quick start (local)

- Linux (host requires Docker):
  scripts/toolchain/linux/build.sh llvmorg-20.1.8 "$PWD/llvm-install" "X86;AArch64"

  # Archive will be written next to the install dir (in the install dir on Linux via /out mount).

- macOS:
  scripts/toolchain/macos/build.sh llvmorg-20.1.8 "$PWD/llvm-install" "X86;AArch64"

  # Archive is created in the current working directory; install goes to the given prefix.

- Windows:
  scripts/toolchain/windows/build.sh llvmorg-20.1.8 "$PWD/llvm-install" "X86;AArch64"
  # Archive is created in the current working directory; install goes to the given prefix.

Notes

- Linux build uses manylinux_2_28 container images (x86_64 or aarch64) and optionally applies BOLT optimization using perf profiles if available.
- macOS build targets macOS 11+ and uses ThinLTO, with optional ccache.
- Windows build uses bash (Git Bash) and performs PGO + ThinLTO similar to macOS (no BOLT step).
- The composite action can be used independently inside reusable workflows to fetch prebuilt assets from GitHub Releases when available or fall back to building.
