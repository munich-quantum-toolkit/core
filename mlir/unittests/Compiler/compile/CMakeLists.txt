# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_test(
  NAME mqt-core-compile-qir-output-test
  COMMAND
    ${CMAKE_COMMAND}
    "-DMQT_CORE_CLI=$<TARGET_FILE:mqt-core>"
    "-DLLVM_AS=$<TARGET_FILE:llvm-as>"
    "-DLLVM_DIS=$<TARGET_FILE:llvm-dis>"
    "-DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/bell.qasm"
    "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/output-$<CONFIG>"
    -P
    "${CMAKE_CURRENT_SOURCE_DIR}/verify_qir_output.cmake")

add_test(
  NAME mqt-core-compile-invalid-mlir-test
  COMMAND
    ${CMAKE_COMMAND} "-DMQT_CORE_CLI=$<TARGET_FILE:mqt-core>"
    "-DNONLINEAR_QCO_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/nonlinear.qco.mlir"
    "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/invalid-output-$<CONFIG>" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/verify_invalid_mlir.cmake")

add_test(
  NAME mqt-core-compile-debugging-test
  COMMAND
    ${CMAKE_COMMAND}
    "-DMQT_CORE_CLI=$<TARGET_FILE:mqt-core>"
    "-DQASM_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/bell.qasm"
    "-DNONLINEAR_QCO_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/nonlinear.qco.mlir"
    "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/debugging-output-$<CONFIG>"
    -P
    "${CMAKE_CURRENT_SOURCE_DIR}/verify_debugging.cmake")

target_sources(mqt-core-mlir-unittests-compiler PRIVATE test_cli_options.cpp)
target_compile_definitions(
  mqt-core-mlir-unittests-compiler
  PRIVATE
    MQT_CORE_CLI="$<TARGET_FILE:mqt-core>"
    MQT_CORE_CLI_INPUT="${CMAKE_CURRENT_SOURCE_DIR}/bell.qasm"
    MQT_CORE_CLI_LAYOUT_INPUT="${CMAKE_CURRENT_SOURCE_DIR}/layout.qc.mlir")
add_dependencies(mqt-core-mlir-unittests-compiler mqt-core)

add_test(
  NAME mqt-core-execution-test
  COMMAND
    ${CMAKE_COMMAND} "-DMQT_CORE_CLI=$<TARGET_FILE:mqt-core>"
    "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/execution-$<CONFIG>" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/verify_execution.cmake")
