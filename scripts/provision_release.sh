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
if [[ $(uname -s) == Linux ]]; then
  uv tool install "sccache>=0.10.0"
  sdk=/opt/llvm-23.1.1
  if [[ ! -x "$sdk/bin/llvm-config" ]]; then
    curl --fail --location --retry 3 \
      https://raw.githubusercontent.com/munich-quantum-software/setup-mlir/01969745aad746dc11e47941c35316704158aa6f/installation/setup-mlir.sh \
      -o "$root/setup-mlir.sh"
    bash "$root/setup-mlir.sh" -v 23.1.1 -p "$sdk" -a OFF
  fi
  [[ $("$sdk/bin/llvm-config" --version) == 23.1.1 ]] || { echo 'Expected LLVM 23.1.1' >&2; exit 1; }
  [[ $("$sdk/bin/llvm-config" --assertion-mode) == OFF ]] || { echo 'Expected assertion-free LLVM' >&2; exit 1; }
fi
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
