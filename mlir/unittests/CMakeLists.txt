# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(programs)
add_subdirectory(Compiler)
add_subdirectory(Dialect)
add_subdirectory(Conversion)

add_custom_target(mqt-core-mlir-unittests)

add_dependencies(
  mqt-core-mlir-unittests
  mqt-core-mlir-compiler-pipeline-test
  mqt-core-mlir-qco-dialect-test
  mqt-core-mlir-dialect-qco-ir-modifiers-test
  mqt-core-mlir-dialect-utils-test
  mqt-core-mlir-unittest-qc-ir
  mqt-core-mlir-unittest-qco-ir
  mqt-core-mlir-unittest-qc-to-qco
  mqt-core-mlir-unittest-qco-to-qc)
