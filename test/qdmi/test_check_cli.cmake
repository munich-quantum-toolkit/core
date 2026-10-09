# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(MAKE_DIRECTORY "${WORK_DIR}")
set(configuration "${WORK_DIR}/devices.json")
foreach(status idle busy offline hang hang-exit)
  file(
    WRITE "${configuration}"
    "{\"schema-version\":1,\"qdmi\":{\"devices\":[{\"id\":\"test.check\",\"library\":\"${SESSION_DEVICE}\",\"prefix\":\"TEST_SESSION\",\"session\":{\"custom4\":\"${status}\"}}]}}"
  )
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env --unset=MQT_CORE_QDMI_DRIVER --unset=MQT_CORE_QDMI_CONFIG_JSON
      "MQT_CORE_QDMI_CONFIG_FILE=${configuration}" "${CHECKER}" --device test.check --timeout 1
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
    TIMEOUT 5)
  if(status MATCHES "^hang")
    set(expected 124)
  elseif(status STREQUAL "offline")
    set(expected 1)
  else()
    set(expected 0)
  endif()
  if(NOT result STREQUAL "${expected}" OR NOT output STREQUAL "")
    message(
      FATAL_ERROR
        "Availability check for ${status}: expected ${expected}, got ${result}: ${output}${error}")
  endif()
endforeach()

foreach(arguments IN
        ITEMS "0;--help" "2;--device" "2;--timeout;1" "2;--device;test.check;--timeout;0"
              "2;--unknown;value" "1;--device;unknown")
  list(POP_FRONT arguments expected)
  execute_process(
    COMMAND "${CHECKER}" ${arguments}
    RESULT_VARIABLE result
    OUTPUT_QUIET ERROR_QUIET
    TIMEOUT 5)
  if(NOT result STREQUAL "${expected}")
    message(FATAL_ERROR "Availability command ${arguments}: expected ${expected}, got ${result}")
  endif()
endforeach()
