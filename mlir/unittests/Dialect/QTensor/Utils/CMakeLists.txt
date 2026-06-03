# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 MQSC GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(qtensor_utils_target mqt-core-mlir-unittest-qtensor-utils)
add_executable(${qtensor_utils_target} test_tensoriterator.cpp)
target_link_libraries(${qtensor_utils_target} PRIVATE GTest::gtest_main MLIRQTensorDialect
                                                      MLIRQTensorUtils MLIRQCOProgramBuilder)
mqt_mlir_configure_unittest_target(${qtensor_utils_target})

gtest_discover_tests(${qtensor_utils_target} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT
                                                        60)
