# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-mlir-unittests-compiler
               test_compiler_pipeline.cpp test_compiler_qdmi_adapter.cpp test_compiler_target.cpp)

target_link_libraries(
  mqt-core-mlir-unittests-compiler
  PRIVATE GTest::gtest_main
          MQTCompilerQDMIAdapter
          MQTCompilerTarget
          MQTCompilerPipeline
          MLIRQCOpenQASMTranslation
          MLIRQASMPrograms
          MLIRQCProgramBuilder
          MLIRQIRProgramBuilder
          MLIRParser
          MLIRSupportMQT
          MLIRQCPrograms
          MLIRQCOPrograms
          MLIRQIRPrograms)

mqt_copy_qdmi_runtime(mqt-core-mlir-unittests-compiler MQT::CoreQDMIScDevice
                      MQT::CoreQDMI_DDSIM_Device MQT::CoreQDMINaDevice)
target_compile_definitions(
  mqt-core-mlir-unittests-compiler
  PRIVATE
    MQT_CORE_MLIR_HETEROGENEOUS_SC_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/Inputs/heterogeneous-sc.json"
    MQT_CORE_MLIR_DIRECTIONAL_ONE_WAY_SC_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/Inputs/directional-one-way-sc.json"
    MQT_CORE_MLIR_DIRECTIONAL_TWO_WAY_SC_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/Inputs/directional-two-way-sc.json"
)

mqt_mlir_configure_unittest_target(mqt-core-mlir-unittests-compiler REQUIRES_EH)

gtest_discover_tests(mqt-core-mlir-unittests-compiler PROPERTIES LABELS mqt-mlir-unittests
                                                                 DISCOVERY_TIMEOUT 60)
