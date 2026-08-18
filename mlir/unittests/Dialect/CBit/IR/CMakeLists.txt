# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(cbit_ir_target mqt-core-mlir-unittest-cbit-ir)
add_executable(${cbit_ir_target} test_cbit_ir.cpp)
target_link_libraries(
  ${cbit_ir_target}
  PRIVATE GTest::gtest_main
          MLIRArithDialect
          MLIRCBitDialect
          MLIRFuncDialect
          MLIRParser
          MLIRSupport
          MLIRTransforms)
mqt_mlir_configure_unittest_target(${cbit_ir_target})

gtest_discover_tests(${cbit_ir_target} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
