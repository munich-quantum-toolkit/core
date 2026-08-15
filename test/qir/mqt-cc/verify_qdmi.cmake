# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

function(require_success description)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${description} failed with exit code ${result}:\n${output}${error}")
  endif()
  set(command_output
      "${output}"
      PARENT_SCOPE)
endfunction()

function(require_failure description expected_error)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(result EQUAL 0)
    message(FATAL_ERROR "${description} unexpectedly succeeded:\n${output}${error}")
  endif()
  string(FIND "${output}${error}" "${expected_error}" error_position)
  if(error_position EQUAL -1)
    message(FATAL_ERROR "${description} did not report '${expected_error}':\n${output}${error}")
  endif()
endfunction()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")

require_success("QDMI device listing" "${MQT_CC}" --qdmi-list-devices)
string(FIND "${command_output}" "mqt.ddsim.default" ddsim_position)
if(ddsim_position EQUAL -1)
  message(FATAL_ERROR "QDMI device listing omitted mqt.ddsim.default:\n${command_output}")
endif()

require_failure("unknown QDMI device" "mqt.unknown.device" "${MQT_CC}" "${INPUT_FILE}"
                --qdmi-device=mqt.unknown.device --emit=qco-optimized)

require_failure("invalid QDMI registry configuration" "Failed to discover registered QDMI devices"
                "${MQT_CC}" "--qdmi-config=${OUTPUT_DIR}/missing.json" --qdmi-list-devices)

require_success("DDSIM target compilation" "${MQT_CC}" "${INPUT_FILE}"
                --qdmi-device=mqt.ddsim.default --emit=qco-optimized)
string(FIND "${command_output}" "qco." qco_position)
if(qco_position EQUAL -1)
  message(FATAL_ERROR "DDSIM target compilation did not produce QCO MLIR:\n${command_output}")
endif()
