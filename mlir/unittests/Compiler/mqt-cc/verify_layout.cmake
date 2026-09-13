# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(MAKE_DIRECTORY "${OUTPUT_DIR}")

foreach(format openqasm3 qir-base qir-adaptive jeff)
  execute_process(
    COMMAND "${MQT_CC}" "${INPUT_FILE}" "--emit=${format}"
    RESULT_VARIABLE result
    ERROR_VARIABLE error
    OUTPUT_QUIET)
  if(result EQUAL 0 OR NOT error MATCHES "cannot preserve qubit layout")
    message(FATAL_ERROR "${format} did not reject layout loss: ${error}")
  endif()
  execute_process(
    COMMAND "${MQT_CC}" "${INPUT_FILE}" "--emit=${format}" --discard-layout
            "-o=${OUTPUT_DIR}/layout.${format}"
    RESULT_VARIABLE result
    ERROR_VARIABLE error
    OUTPUT_QUIET)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${format} failed after explicit discard: ${error}")
  endif()
endforeach()

execute_process(
  COMMAND "${MQT_CC}" "${INPUT_FILE}" --emit=qc-import
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT output MATCHES "mqt.layout =")
  message(FATAL_ERROR "Import did not retain layout metadata: ${output}${error}")
endif()

execute_process(
  COMMAND "${MQT_CC}" "${INPUT_FILE}" --emit=qco-optimized
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT output MATCHES "mqt.layout_invalidated")
  message(FATAL_ERROR "Optimization did not invalidate layout: ${output}${error}")
endif()

execute_process(
  COMMAND "${MQT_CC}" "${INPUT_FILE}" --run-pipeline "--pass-pipeline=builtin.module(canonicalize)"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT output MATCHES "mqt.layout_invalidated")
  message(FATAL_ERROR "Custom pipeline did not invalidate layout: ${output}${error}")
endif()
