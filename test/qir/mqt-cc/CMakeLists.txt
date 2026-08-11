# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_test(
  NAME mqt-core-mqt-cc-qir-output-test
  COMMAND
    ${CMAKE_COMMAND} "-DMQT_CC=$<TARGET_FILE:mqt-cc>" "-DLLVM_AS=$<TARGET_FILE:llvm-as>"
    "-DLLVM_DIS=$<TARGET_FILE:llvm-dis>" "-DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/bell.qasm"
    "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/output-$<CONFIG>" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/verify_qir_output.cmake")

add_test(
  NAME mqt-core-mqt-cc-rejects-noncircuit-device-test
  COMMAND $<TARGET_FILE:mqt-cc> "${CMAKE_CURRENT_SOURCE_DIR}/bell.qasm" --emit=qir-base
          --qdmi-device=mqt.na.default -o "${CMAKE_CURRENT_BINARY_DIR}/invalid-device-$<CONFIG>.ll")
set_tests_properties(
  mqt-core-mqt-cc-rejects-noncircuit-device-test
  PROPERTIES PASS_REGULAR_EXPRESSION
             "Failed to create compiler target.*only circuit-model devices.*zone sites")
