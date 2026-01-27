# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-mlir-conversion-jeff-to-qc-test test_conversion_jeff_to_qc.cpp)

target_link_libraries(mqt-core-mlir-conversion-jeff-to-qc-test
                      PRIVATE GTest::gtest_main MLIRParser MLIRPass MLIRTransforms JeffToQC)

gtest_discover_tests(mqt-core-mlir-conversion-jeff-to-qc-test)
