# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(support_target mqt-core-mlir-unittests-support)
add_executable(${support_target} test_constant_folding.cpp)
target_link_libraries(${support_target} PRIVATE GTest::gtest_main MLIRArithDialect MLIRFuncDialect
                                                MLIRIR MLIRSupport)
mqt_mlir_configure_unittest_target(${support_target})

gtest_discover_tests(${support_target} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
