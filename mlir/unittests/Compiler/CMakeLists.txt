# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(compiler_test_sources test_compiler_pipeline.cpp test_compiler_target.cpp)
set(has_compiler_qdmi_provider_tests OFF)
if(TARGET MQT::CoreQDMIScDevice
   AND TARGET MQT::CoreQDMI_DDSIM_Device
   AND TARGET MQT::CoreQDMINaDevice)
  list(APPEND compiler_test_sources test_compiler_fomac_adapter.cpp)
  set(has_compiler_qdmi_provider_tests ON)
endif()

add_executable(mqt-core-mlir-unittests-compiler ${compiler_test_sources})

target_link_libraries(
  mqt-core-mlir-unittests-compiler
  PRIVATE GTest::gtest_main
          MQTCompilerFoMaCAdapter
          MQTCompilerTarget
          MQTCompilerPipeline
          MLIRQCTranslation
          MLIRQASMPrograms
          MLIRQCProgramBuilder
          MLIRQIRProgramBuilder
          MLIRParser
          MLIRSupportMQT
          MLIRQCPrograms
          MLIRQCOPrograms
          MLIRQIRPrograms
          MLIRQuantumComputationPrograms
          MQT::CoreIR)

if(has_compiler_qdmi_provider_tests)
  mqt_copy_qdmi_runtime(mqt-core-mlir-unittests-compiler MQT::CoreQDMIScDevice
                        MQT::CoreQDMI_DDSIM_Device MQT::CoreQDMINaDevice)
  target_compile_definitions(
    mqt-core-mlir-unittests-compiler
    PRIVATE
      MQT_CORE_MLIR_HETEROGENEOUS_SC_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/Inputs/heterogeneous-sc.json"
      MQT_CORE_MLIR_DIRECTIONAL_ONE_WAY_SC_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/Inputs/directional-one-way-sc.json"
      MQT_CORE_MLIR_DIRECTIONAL_TWO_WAY_SC_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/Inputs/directional-two-way-sc.json"
  )
endif()

mqt_mlir_configure_unittest_target(mqt-core-mlir-unittests-compiler)

gtest_discover_tests(mqt-core-mlir-unittests-compiler PROPERTIES LABELS mqt-mlir-unittests
                                                                 DISCOVERY_TIMEOUT 60)

if(TARGET MQT::CoreQDMIScDevice)
  # `mqt-cc` is normally an optional LLVM tool. Build it only as a dependency of the test target
  # when its irreducible subprocess tests are enabled.
  add_dependencies(mqt-core-mlir-unittests-compiler mqt-cc)

  set(mqt_cc_test_input ${CMAKE_CURRENT_SOURCE_DIR}/Inputs/bell.qasm)
  add_test(NAME mqt-cc-qdmi-list COMMAND $<TARGET_FILE:mqt-cc> --qdmi-list-devices)
  set_tests_properties(mqt-cc-qdmi-list PROPERTIES LABELS mqt-mlir-unittests
                                                   PASS_REGULAR_EXPRESSION "mqt.sc.iqm.garnet")

  add_test(
    NAME mqt-cc-qdmi-explicit-config
    COMMAND
      $<TARGET_FILE:mqt-cc>
      "--qdmi-config=${CMAKE_CURRENT_SOURCE_DIR}/Inputs/mqt-cc-explicit.qdmi.json"
      --qdmi-list-devices)
  set_tests_properties(
    mqt-cc-qdmi-explicit-config
    PROPERTIES LABELS mqt-mlir-unittests PASS_REGULAR_EXPRESSION "mqt.sc.iqm.emerald"
               FAIL_REGULAR_EXPRESSION "mqt.sc.iqm.garnet")

  add_test(NAME mqt-cc-qdmi-unknown COMMAND $<TARGET_FILE:mqt-cc> --qdmi-device=unknown
                                            ${mqt_cc_test_input})
  set_tests_properties(
    mqt-cc-qdmi-unknown PROPERTIES LABELS mqt-mlir-unittests PASS_REGULAR_EXPRESSION
                                   "mqt-cc failed: Unknown QDMI device ID 'unknown'")

  add_test(NAME mqt-cc-qdmi-garnet COMMAND $<TARGET_FILE:mqt-cc> --qdmi-device=mqt.sc.iqm.garnet
                                           --emit=qco-optimized ${mqt_cc_test_input})
  set_tests_properties(
    mqt-cc-qdmi-garnet PROPERTIES LABELS mqt-mlir-unittests PASS_REGULAR_EXPRESSION "qco.r"
                                  FAIL_REGULAR_EXPRESSION "qco.rx|qco.ry")
endif()
