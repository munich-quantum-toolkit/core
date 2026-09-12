#!/usr/bin/env bash
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -euo pipefail

root=${1:?release tool directory}
mkdir -p "$root/sdk-tools" "$root/llvm-source"
sdk_revision=57fd0184f2fd416effcc131eeeff17659c07c7fb
llvm_revision=6dfe1677ab8dffbc6ec13d53a1e0215d75147689
curl --fail --location --retry 3 \
  "https://github.com/munich-quantum-software/portable-mlir-toolchain/archive/$sdk_revision.tar.gz" \
  -o "$root/sdk-tools.tar.gz"
tar -xf "$root/sdk-tools.tar.gz" -C "$root/sdk-tools" --strip-components=1
curl --fail --location --retry 3 \
  "https://github.com/llvm/llvm-project/archive/$llvm_revision.tar.gz" -o "$root/llvm-source.tar.gz"
tar -xf "$root/llvm-source.tar.gz" -C "$root/llvm-source" --strip-components=1
shasum -a 256 "$root/sdk-tools.tar.gz" "$root/llvm-source.tar.gz" > "$root/sources.sha256"
printf '%s\n' "$llvm_revision" > "$root/llvm-revision"
rm "$root/sdk-tools.tar.gz" "$root/llvm-source.tar.gz"

if [[ $(uname -s) == Linux ]]; then
  manylinux-install-clang -v 22.1.8.1 -c 8b399744aeb49c70048b379b9b3ffc651d86fde808551c8cc4138c4fadc5308e
  python "$root/sdk-tools/scripts/toolchain/linux/configure-linker-stack.py" /opt/clang/bin/lld \
    > "$root/linker-stack.json"
  export CC=/opt/clang/bin/clang CXX=/opt/clang/bin/clang++
  export AR=/opt/clang/bin/llvm-ar RANLIB=/opt/clang/bin/llvm-ranlib
  export PATH="/opt/clang/bin:$PATH"
  bash "$root/sdk-tools/scripts/toolchain/linux/install-profile-tools.sh" "$root/profiling"
fi
