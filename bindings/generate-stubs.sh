#!/bin/bash
# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -euo pipefail

core_dir=./python/mqt/core
stub_files=(
  "$core_dir/ir/__init__.pyi"
  "$core_dir/ir/operations.pyi"
  "$core_dir/ir/registers.pyi"
  "$core_dir/ir/symbolic.pyi"
  "$core_dir/dd.pyi"
  "$core_dir/fomac.pyi"
  "$core_dir/na/fomac.pyi"
)

core_patterns=./bindings/core_patterns.txt

# Check if all stub files exist
for stub_file in "${stub_files[@]}"; do
  if [ ! -f "$stub_file" ]; then
    echo "Error: $stub_file does not exist. Are you running this script from the root directory?"
    exit 1
  fi
done

if [ ! -f "$core_patterns" ]; then
  echo "Error: $core_patterns does not exist. Are you running this script from the root directory?"
  exit 1
fi

# Ensure that the most recent version of mqt.core is installed
uv sync

# Remove the existing stub files
for stub_file in "${stub_files[@]}"; do
  rm -f "$stub_file"
done

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
for stub_file in "${stub_files[@]}"; do
  uvx prek license-tools --quiet --files "$stub_file"
  uvx prek ruff-check --quiet --files "$stub_file"
  uvx prek ruff-format --quiet --files "$stub_file"
done
