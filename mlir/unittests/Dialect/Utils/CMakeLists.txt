# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-mlir-unittests-dialect-utils test_utils.cpp)

target_link_libraries(mqt-core-mlir-unittests-dialect-utils
                      PRIVATE GTest::gtest_main MLIRArithDialect MLIRIR MLIRSupport LLVMSupport)

mqt_mlir_configure_unittest_target(mqt-core-mlir-unittests-dialect-utils)

gtest_discover_tests(mqt-core-mlir-unittests-dialect-utils PROPERTIES LABELS mqt-mlir-unittests
                                                                      DISCOVERY_TIMEOUT 60)
