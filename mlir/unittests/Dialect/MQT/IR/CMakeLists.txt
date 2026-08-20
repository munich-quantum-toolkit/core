# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(mqt_ir_target mqt-core-mlir-unittest-mqt-ir)
add_executable(${mqt_ir_target} test_mqt_ir.cpp)
target_link_libraries(
  ${mqt_ir_target}
  PRIVATE GTest::gtest_main
          MLIRArithDialect
          MLIRFuncDialect
          MLIRMemRefDialect
          MLIRMQTDialect
          MLIRParser
          MLIRQCDialect
          MLIRQCODialect
          MLIRQTensorDialect)
mqt_mlir_configure_unittest_target(${mqt_ir_target})

gtest_discover_tests(${mqt_ir_target} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
