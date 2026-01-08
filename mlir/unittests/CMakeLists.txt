# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(dialect)
add_subdirectory(pipeline)
add_subdirectory(translation)

add_custom_target(mqt-core-mlir-unittests)

add_dependencies(mqt-core-mlir-unittests mqt-core-mlir-compiler-pipeline-test
                 mqt-core-mlir-translation-test mqt-core-mlir-wireiterator-test)
