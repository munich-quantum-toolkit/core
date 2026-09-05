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
set(qdmi_test_config "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/qdmi-adapter-devices.json")
file(
  GENERATE
  OUTPUT "${qdmi_test_config}"
  CONTENT
    "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"test.mlir.heterogeneous\", \"library\": \"$<TARGET_FILE:MQT::CoreQDMIScDevice>\", \"prefix\": \"MQT_SC\", \"session\": {\"device-config\": {\"file\": \"${CMAKE_CURRENT_SOURCE_DIR}/Inputs/heterogeneous-sc.json\"}}},\n      {\"id\": \"test.mlir.higher-arity\", \"library\": \"$<TARGET_FILE:MQT::CoreQDMIScDevice>\", \"prefix\": \"MQT_SC\", \"session\": {\"device-config\": {\"file\": \"${CMAKE_CURRENT_SOURCE_DIR}/Inputs/higher-arity-sc.json\"}}},\n      {\"id\": \"test.mlir.directional-one-way\", \"library\": \"$<TARGET_FILE:MQT::CoreQDMIScDevice>\", \"prefix\": \"MQT_SC\", \"session\": {\"device-config\": {\"file\": \"${CMAKE_CURRENT_SOURCE_DIR}/Inputs/directional-one-way-sc.json\"}}},\n      {\"id\": \"test.mlir.directional-two-way\", \"library\": \"$<TARGET_FILE:MQT::CoreQDMIScDevice>\", \"prefix\": \"MQT_SC\", \"session\": {\"device-config\": {\"file\": \"${CMAKE_CURRENT_SOURCE_DIR}/Inputs/directional-two-way-sc.json\"}}},\n      {\"id\": \"test.mlir.unavailable-operation\", \"library\": \"$<TARGET_FILE:MQT::CoreQDMIScDevice>\", \"prefix\": \"MQT_SC\", \"session\": {\"device-config\": {\"file\": \"${CMAKE_CURRENT_SOURCE_DIR}/Inputs/unavailable-operation-sc.json\"}}}\n    ]\n  }\n}\n"
)
target_compile_definitions(mqt-core-mlir-unittests-compiler
                           PRIVATE MQT_CORE_MLIR_QDMI_TEST_CONFIG="${qdmi_test_config}")

mqt_mlir_configure_unittest_target(mqt-core-mlir-unittests-compiler REQUIRES_EH)

gtest_discover_tests(mqt-core-mlir-unittests-compiler PROPERTIES LABELS mqt-mlir-unittests
                                                                 DISCOVERY_TIMEOUT 60)

add_subdirectory(mqt-cc)
