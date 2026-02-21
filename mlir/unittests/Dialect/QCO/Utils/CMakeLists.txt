# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(qco_utils_target mqt-core-mlir-unittest-qco-utils)
add_executable(${qco_utils_target} test_wireiterator.cpp)
target_link_libraries(${qco_utils_target} PRIVATE GTest::gtest_main MLIRQCODialect
                                                  MLIRQCOProgramBuilder)
mqt_mlir_configure_unittest_target(${qco_utils_target})

gtest_discover_tests(${qco_utils_target} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
