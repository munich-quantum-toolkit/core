# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Build the compiler driver executable
add_mlir_tool(mqt-cc mqt-cc.cpp DEPENDS MQTCompilerPipeline SUPPORT_PLUGINS)
llvm_map_components_to_libnames(llvm_native_libs bitwriter)
target_link_libraries(
  mqt-cc
  PRIVATE MQTCompilerPipeline
          MLIRParser
          MLIRSupport
          MLIRQCTranslation
          MLIRJeffTranslation
          MLIRJeffToQCO
          MLIRBytecodeWriter
          MLIRTargetLLVMIRExport
          MLIRBuiltinToLLVMIRTranslation
          MLIRLLVMToLLVMIRTranslation
          ${llvm_native_libs})

mqt_mlir_target_use_project_options(mqt-cc)
llvm_update_compile_flags(mqt-cc)
mlir_check_all_link_libraries(mqt-cc)
export_executable_symbols_for_plugins(mqt-cc)

if(BUILD_MQT_CORE_TESTS)
  set_target_properties(mqt-cc PROPERTIES EXCLUDE_FROM_ALL FALSE)

  add_test(NAME mqt-cc-coupling-map-qco-optimized
           COMMAND $<TARGET_FILE:mqt-cc> ${CMAKE_CURRENT_SOURCE_DIR}/tests/coupling-line.qasm
                   --emit=qco-optimized --native-gates=u,cx --coupling-map=0-1,1-2)
  set_tests_properties(
    mqt-cc-coupling-map-qco-optimized
    PROPERTIES LABELS mqt-mlir-unittests PASS_REGULAR_EXPRESSION "qco\\.ctrl"
               FAIL_REGULAR_EXPRESSION "qco\\.swap;qco\\.ctrl\\(%0\\).*targets.*%2")

  add_test(
    NAME mqt-cc-coupling-map-requires-native-gates
    COMMAND
      ${CMAKE_COMMAND} -DMQT_CC=$<TARGET_FILE:mqt-cc>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/coupling-line.qasm -P
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/check-coupling-map-requires-native-gates.cmake)
  set_tests_properties(mqt-cc-coupling-map-requires-native-gates PROPERTIES LABELS
                                                                            mqt-mlir-unittests)
endif()
