# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(target_name mqt-core-mlir-unittest-qc-ir)
add_executable(${target_name} test_qc_ir.cpp test_qc_modifier_simplification.cpp
               test_qc_numeric_simplification.cpp)
target_link_libraries(
  ${target_name}
  PRIVATE
    MLIRParser
    MLIRSupportMQT
    GTest::gtest_main
    MLIRMQTDialect
    MLIRQCProgramBuilder
    MLIRQCPrograms
    MLIRQCToQCO
    MLIRQCOUtils
    MLIRQCODDFunctionality
    MLIRExactUnitaryTest
    MLIRMQTTransforms)
mqt_mlir_configure_unittest_target(${target_name} REQUIRES_EH)
