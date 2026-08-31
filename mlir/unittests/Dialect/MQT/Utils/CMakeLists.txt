# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(mqt_utils_target mqt-core-mlir-unittests-mqt-utils)
add_executable(${mqt_utils_target} test_constant_folding.cpp test_gate_powering.cpp)
target_link_libraries(
  ${mqt_utils_target} PRIVATE GTest::gtest_main MLIRArithDialect MLIRFuncDialect MLIRIndexDialect
                              MLIRIR MLIRMQTUtils)
mqt_mlir_configure_unittest_target(${mqt_utils_target})

gtest_discover_tests(${mqt_utils_target} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
