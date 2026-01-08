# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-mlir-wireiterator-test test_wireiterator.cpp)

target_link_libraries(
  mqt-core-mlir-wireiterator-test PRIVATE GTest::gtest_main MLIRParser MLIRMQTOpt MLIRSCFDialect
                                          MLIRArithDialect MLIRIndexDialect)

gtest_discover_tests(mqt-core-mlir-wireiterator-test)
