# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(target_name mqt-core-mlir-unittest-target-synthesis)
add_executable(${target_name} test_target_synthesis.cpp)

target_link_libraries(
  ${target_name}
  PRIVATE MLIRParser
          GTest::gtest_main
          MQTCompilerTarget
          MLIRQCOProgramBuilder
          MLIRQCODDFunctionality
          MLIRQCOUtils
          MLIRQCOTransforms
          MLIRPass
          MLIRFuncDialect
          MLIRArithDialect
          MLIRSCFDialect
          MLIRIR
          MLIRSupport
          LLVMSupport)

mqt_mlir_configure_unittest_target(${target_name} REQUIRES_EH)

gtest_discover_tests(${target_name} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
