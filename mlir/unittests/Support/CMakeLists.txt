# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-mlir-unittests-support test_ir_verification.cpp)
target_link_libraries(
  mqt-core-mlir-unittests-support
  PRIVATE GTest::gtest_main
          MLIRParser
          MLIRArithDialect
          MLIRControlFlowDialect
          MLIRFuncDialect
          MLIRLLVMDialect
          MLIRQCOUtils)
mqt_mlir_configure_unittest_target(mqt-core-mlir-unittests-support)
gtest_discover_tests(mqt-core-mlir-unittests-support PROPERTIES LABELS mqt-mlir-unittests)
