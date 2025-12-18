#!/bin/bash
# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -euo pipefail

files=(
  "./python/mqt/core/ir/__init__.pyi"
  "./python/mqt/core/ir/operations.pyi"
  "./python/mqt/core/ir/registers.pyi"
  "./python/mqt/core/ir/symbolic.pyi"
  "./python/mqt/core/dd.pyi"
  "./python/mqt/core/fomac.pyi"
  "./python/mqt/core/na/fomac.pyi"
)

# Check if all files exist
for file in "${files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "Error: $file does not exist. Are you running this script from the root directory?"
    exit 1
  fi
done

# Ensure that the most recent version of mqt.core is installed
uv sync

# Remove the existing stub files
for file in "${files[@]}"; do
  rm -f "$file"
done

# Define common paths
core_dir=./python/mqt/core
core_patterns=./bindings/core_patterns.txt

# Generate new stub files
uv run --no-sync -m nanobind.stubgen -m mqt.core.ir -o "$core_dir/ir/__init__.pyi" -p "$core_patterns" -P
uv run --no-sync -m nanobind.stubgen -m mqt.core.ir.operations -o "$core_dir/ir/operations.pyi" -p "$core_patterns" -P
uv run --no-sync -m nanobind.stubgen -m mqt.core.ir.registers -o "$core_dir/ir/registers.pyi" -p "$core_patterns" -P
uv run --no-sync -m nanobind.stubgen -m mqt.core.ir.symbolic -o "$core_dir/ir/symbolic.pyi" -p "$core_patterns" -P
uv run --no-sync -m nanobind.stubgen -m mqt.core.dd -o "$core_dir/dd.pyi" -p "$core_patterns" -P
uv run --no-sync -m nanobind.stubgen -m mqt.core.fomac -o "$core_dir/fomac.pyi" -p "$core_patterns" -P
uv run --no-sync -m nanobind.stubgen -m mqt.core.na.fomac -o "$core_dir/na/fomac.pyi" -p "$core_patterns" -P

set +e

# Run prek on generated stub files
for file in "${files[@]}"; do
  uvx prek --files "$file"
done
