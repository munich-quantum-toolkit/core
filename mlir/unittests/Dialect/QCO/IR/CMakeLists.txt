# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-mlir-qco-dialect-test test_unitary_matrix.cpp)

target_link_libraries(mqt-core-mlir-qco-dialect-test PRIVATE GTest::gtest_main MLIRQCODialect
                                                             MLIRLLVMDialect MLIRQCOProgramBuilder)

gtest_discover_tests(mqt-core-mlir-qco-dialect-test)

add_subdirectory(Modifiers)
