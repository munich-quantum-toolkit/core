# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(MAKE_DIRECTORY "${WORK_DIR}")
set(configuration "${WORK_DIR}/devices.json")
set(manifest "${WORK_DIR}/other-device.qdmi.json")
file(
  WRITE "${manifest}"
  "{\"schema-version\":1,\"qdmi\":{\"devices\":[{\"id\":\"test.unrelated\",\"library\":\"${SESSION_DEVICE}\",\"prefix\":\"TEST_SESSION\",\"session\":{\"custom4\":\"hang\"}}]}}"
)
foreach(status idle busy offline hang hang-exit)
  file(
    WRITE "${configuration}"
    "{\"schema-version\":1,\"qdmi\":{\"devices\":[{\"id\":\"test.check\",\"library\":\"${SESSION_DEVICE}\",\"prefix\":\"TEST_SESSION\",\"session\":{\"custom4\":\"${status}\"}},{\"id\":\"test.unrelated\",\"enabled\":false}]}}"
  )
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env --unset=MQT_CORE_QDMI_DRIVER --unset=MQT_CORE_QDMI_CONFIG_JSON
      "MQT_CORE_QDMI_CONFIG_FILE=${configuration}" "${CHECKER}" --manifest
      "${WORK_DIR}/missing.qdmi.json" --manifest "${manifest}" --device test.check --timeout 1
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

file(REMOVE "${WORK_DIR}/child-started" "${WORK_DIR}/child-survived")
file(
  WRITE "${WORK_DIR}/child.cmake"
  "file(WRITE \"${WORK_DIR}/child-started\" \"\")\n"
  "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E sleep 3)\n"
  "file(WRITE \"${WORK_DIR}/child-survived\" \"\")\n")
set(child_command "\"${CMAKE_COMMAND}\" -P \"${WORK_DIR}/child.cmake\"")
if(WIN32)
  set(child_command "\"${child_command}\"")
endif()
string(REPLACE "\"" "\\\"" child_command "${child_command}")
file(
  WRITE "${configuration}"
  "{\"schema-version\":1,\"qdmi\":{\"devices\":[{\"id\":\"test.check\",\"library\":\"${SESSION_DEVICE}\",\"prefix\":\"TEST_SESSION\",\"session\":{\"custom4\":\"hang-child\",\"custom5\":\"${child_command}\"}}]}}"
)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env --unset=MQT_CORE_QDMI_DRIVER --unset=MQT_CORE_QDMI_CONFIG_JSON
          "MQT_CORE_QDMI_CONFIG_FILE=${configuration}" "${CHECKER}" --device test.check --timeout 2
  RESULT_VARIABLE result
  OUTPUT_QUIET ERROR_QUIET
  TIMEOUT 8)
if(NOT result STREQUAL "124" OR NOT EXISTS "${WORK_DIR}/child-started")
  message(FATAL_ERROR "The descendant cleanup test did not reach its timeout: ${result}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 2)
if(EXISTS "${WORK_DIR}/child-survived")
  message(FATAL_ERROR "A device subprocess survived the availability timeout")
endif()

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
