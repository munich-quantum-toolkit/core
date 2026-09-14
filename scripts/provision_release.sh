#!/usr/bin/env bash
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -euo pipefail

root=${1:?project directory}/build/release-tools
mkdir -p "$root/llvm-source"
if [[ $(uname -s) == Linux ]]; then
  dnf install -y clang llvm compiler-rt lld
  uv tool install sccache
  if [[ ! -x /opt/llvm/bin/llvm-config ]]; then
    curl --fail --location --retry 3 \
      https://raw.githubusercontent.com/munich-quantum-software/setup-mlir/main/installation/setup-mlir.sh \
      -o "$root/setup-mlir.sh"
    bash "$root/setup-mlir.sh" -v 23.1.1 -p /opt/llvm -a OFF
  fi
fi
version=$("${MLIR_DIR:?}/../../../bin/llvm-config" --version)
curl --fail --location --retry 3 \
  "https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-$version.tar.gz" \
  | tar -xz -C "$root/llvm-source" --strip-components=1
