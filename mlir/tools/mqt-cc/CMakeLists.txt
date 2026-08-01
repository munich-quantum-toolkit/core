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
          MLIRMathDialect
          MLIRTargetLLVMIRExport
          MLIRBuiltinToLLVMIRTranslation
          MLIRLLVMToLLVMIRTranslation
          ${llvm_native_libs})

mqt_mlir_target_use_project_options(mqt-cc)
llvm_update_compile_flags(mqt-cc)
mlir_check_all_link_libraries(mqt-cc)
export_executable_symbols_for_plugins(mqt-cc)

if(BUILD_MQT_CORE_TESTS)
  set(mqt_cc_reuse_qubits_input ${CMAKE_CURRENT_SOURCE_DIR}/test/reuse-qubits.mlir)
  set(mqt_cc_reuse_qubits_full_input ${CMAKE_CURRENT_SOURCE_DIR}/test/reuse-qubits-full.mlir)
  add_test(NAME mqt-cc-reuse-qubits COMMAND mqt-cc ${mqt_cc_reuse_qubits_input}
                                            --emit=qco-optimized --reuse-qubits)
  add_test(NAME mqt-cc-reuse-qubits-full COMMAND mqt-cc ${mqt_cc_reuse_qubits_full_input}
                                                 --emit=qco-optimized --reuse-qubits-full)
  add_test(
    NAME mqt-cc-reuse-qubits-mutually-exclusive
    COMMAND
      ${CMAKE_COMMAND} -DMQT_CC_EXECUTABLE=$<TARGET_FILE:mqt-cc>
      -DTEST_INPUT=${mqt_cc_reuse_qubits_input} -DTEST_CASE=mutually-exclusive -P
      ${CMAKE_CURRENT_SOURCE_DIR}/test/expect-reuse-qubits-option-error.cmake)
  add_test(
    NAME mqt-cc-reuse-qubits-custom-pipeline
    COMMAND
      ${CMAKE_COMMAND} -DMQT_CC_EXECUTABLE=$<TARGET_FILE:mqt-cc>
      -DTEST_INPUT=${mqt_cc_reuse_qubits_input} -DTEST_CASE=custom-pipeline -P
      ${CMAKE_CURRENT_SOURCE_DIR}/test/expect-reuse-qubits-option-error.cmake)
  set_tests_properties(mqt-cc-reuse-qubits mqt-cc-reuse-qubits-full
                       PROPERTIES LABELS mqt-mlir-unittests PASS_REGULAR_EXPRESSION "qco.reset")
  set_tests_properties(mqt-cc-reuse-qubits-mutually-exclusive mqt-cc-reuse-qubits-custom-pipeline
                       PROPERTIES LABELS mqt-mlir-unittests)
endif()
