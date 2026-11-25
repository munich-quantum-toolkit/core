# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(target_name mqt-core-mlir-unittest-mapping)
add_executable(${target_name} test_mapping.cpp)

target_link_libraries(
  ${target_name}
  PRIVATE
    GTest::gtest_main
    MLIRParser
    MLIRMQTDialect
    MQTCompilerTarget
    MQTCompilerPipeline
    MLIRQCODDFunctionality
    MLIRQCOProgramBuilder
    MLIRQTensorUtils
    MLIRQCOTransforms
    MLIRSupportMQT)

mqt_mlir_configure_unittest_target(${target_name})
