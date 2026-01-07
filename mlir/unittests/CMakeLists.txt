# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(decomposition)
add_subdirectory(dialect)
add_subdirectory(pipeline)
add_subdirectory(translation)

add_custom_target(mqt-core-mlir-unittests)

add_dependencies(
  mqt-core-mlir-unittests mqt-core-mlir-decomposition-test mqt-core-mlir-compiler-pipeline-test
  mqt-core-mlir-translation-test mqt-core-mlir-wireiterator-test)
