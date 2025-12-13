#!/bin/bash
# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

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
for file in ${files[@]}; do
  if [ ! -f "$file" ]; then
    echo "Error: $file does not exist. Are you running this script from the root directory?"
    exit 1
  fi
done

# Ensure that the most recent version of mqt.core is installed
uv sync

# Remove the existing stub files
for file in ${files[@]}; do
  rm -f "$file"
done

# Generate new stub files
python -m nanobind.stubgen -m mqt.core.ir -o python/mqt/core/ir/__init__.pyi -P
python -m nanobind.stubgen -m mqt.core.ir.operations -o python/mqt/core/ir/operations.pyi -P
python -m nanobind.stubgen -m mqt.core.ir.registers -o python/mqt/core/ir/registers.pyi -P
python -m nanobind.stubgen -m mqt.core.ir.symbolic -o python/mqt/core/ir/symbolic.pyi -P
python -m nanobind.stubgen -m mqt.core.dd -o python/mqt/core/dd.pyi -P
python -m nanobind.stubgen -m mqt.core.fomac -o python/mqt/core/fomac.pyi -P
python -m nanobind.stubgen -m mqt.core.na.fomac -o python/mqt/core/na/fomac.pyi -P

# Remove private Enum members from the stub files
for file in ${files[@]}; do
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' '/_hashable_values_:/d' "$file"
    sed -i '' '/_unhashable_values_map_:/d' "$file"
  else
    sed -i '/_hashable_values_:/d' "$file"
    sed -i '/_unhashable_values_map_:/d' "$file"
  fi
done

# Add license headers to the generated stub files
for file in ${files[@]}; do
  prek license-tools --files "$file"
done

# Run ruff twice to ensure all formatting issues are resolved
uvx ruff format
uvx ruff check

uvx ruff format
uvx ruff check
