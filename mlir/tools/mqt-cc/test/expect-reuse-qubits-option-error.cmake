# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TEST_CASE STREQUAL "mutually-exclusive")
  set(test_arguments --reuse-qubits --reuse-qubits-full)
  set(expected_error "mutually exclusive")
elseif(TEST_CASE STREQUAL "custom-pipeline")
  set(test_arguments --reuse-qubits "--pass-pipeline=builtin.module(canonicalize)")
  set(expected_error "cannot be combined with custom pass options")
else()
  message(FATAL_ERROR "Unknown reuse-qubits option error test case: ${TEST_CASE}")
endif()

execute_process(
  COMMAND "${MQT_CC_EXECUTABLE}" "${TEST_INPUT}" --emit=qco-optimized ${test_arguments}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)

if(result EQUAL 0)
  message(FATAL_ERROR "mqt-cc unexpectedly accepted ${TEST_CASE}")
endif()

string(CONCAT command_output "${output}" "${error}")
if(NOT command_output MATCHES "${expected_error}")
  message(
    FATAL_ERROR
      "mqt-cc failed without the expected '${expected_error}' diagnostic:\n${command_output}")
endif()
