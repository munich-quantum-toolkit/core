# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-mlir-unittests-compiler
               test_compiler_pipeline.cpp test_compiler_qdmi_adapter.cpp test_compiler_target.cpp)

set_target_properties(mqt-core-mlir-unittests-compiler PROPERTIES UNITY_BUILD OFF)

target_link_libraries(
  mqt-core-mlir-unittests-compiler
  PRIVATE GTest::gtest_main
          MLIRExactUnitaryTest
          MQTCompilerQDMIAdapter
          MQTCompilerTarget
          MQTCompilerPipeline
          MLIRCBitDialect
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
                      MQT::CoreQDMI_DDSIM_Device)
target_compile_definitions(mqt-core-mlir-unittests-compiler
                           PRIVATE MQT_CORE_MLIR_SC_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Inputs")
target_link_libraries(mqt-core-mlir-unittests-compiler PRIVATE nlohmann_json::nlohmann_json)

mqt_mlir_configure_unittest_target(mqt-core-mlir-unittests-compiler REQUIRES_EH)

gtest_discover_tests(mqt-core-mlir-unittests-compiler PROPERTIES LABELS mqt-mlir-unittests
                                                                 DISCOVERY_TIMEOUT 60)

add_subdirectory(mqt-cc)
