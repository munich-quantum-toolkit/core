# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(MAKE_DIRECTORY "${WORK_DIR}")
set(configuration "${WORK_DIR}/devices.json")
set(marker "${WORK_DIR}/worker.pid")

function(check_exit expected)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env --unset=MQT_CORE_QDMI_CONFIG_JSON
            "MQT_CORE_QDMI_CONFIG_FILE=${configuration}" "${CHECKER}" ${ARGN}
    RESULT_VARIABLE actual
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
    TIMEOUT 10)
  if(NOT actual STREQUAL "${expected}")
    message(FATAL_ERROR "Expected exit ${expected}, got ${actual}: ${error}")
  endif()
  if(expected EQUAL 1 AND NOT error STREQUAL "QDMI device check failed\n")
    message(FATAL_ERROR "Unexpected failure diagnostic: ${error}")
  endif()
  if(expected EQUAL 124 AND NOT error STREQUAL "QDMI device check timed out\n")
    message(FATAL_ERROR "Unexpected timeout diagnostic: ${error}")
  endif()
  if(NOT output STREQUAL "")
    message(FATAL_ERROR "The checker leaked worker output: ${output}")
  endif()
endfunction()

foreach(
  arguments
  ""
  "--device"
  "--unknown;x"
  "--device;x;--timeout;0"
  "--device;x;--timeout;no"
  "--device;x;--timeout;3601"
  "--device;x;--device;y")
  check_exit(2 ${arguments})
endforeach()

foreach(
  status
  idle
  busy
  offline
  fail-init
  fail-query
  crash
  hang
  hang-free
  descendant)
  file(
    WRITE "${configuration}"
    "{\"schema-version\":1,\"qdmi\":{\"devices\":[{\"id\":\"test.check\",\"library\":\"${SESSION_DEVICE}\",\"prefix\":\"TEST_SESSION\",\"session\":{\"custom4\":\"${status}\",\"custom3\":\"${marker}\"}}]}}"
  )
  if(status STREQUAL "idle" OR status STREQUAL "busy")
    check_exit(0 --device test.check)
  elseif(status STREQUAL "descendant")
    file(REMOVE "${marker}")
    check_exit(0 --device test.check)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1)
    if(EXISTS "${marker}")
      message(FATAL_ERROR "A provider descendant survived checker exit")
    endif()
  elseif(status MATCHES "^hang")
    file(REMOVE "${marker}")
    check_exit(124 --device test.check --timeout 1)
    file(READ "${marker}" worker)
    string(STRIP "${worker}" worker)
    execute_process(
      COMMAND /bin/kill -0 "${worker}"
      RESULT_VARIABLE alive
      ERROR_QUIET)
    if(alive EQUAL 0)
      message(FATAL_ERROR "Checker worker survived its timeout")
    endif()
  else()
    check_exit(1 --device test.check)
  endif()
endforeach()

check_exit(1 --device unregistered)
file(WRITE "${configuration}"
     "{\"schema-version\":1,\"qdmi\":{\"devices\":[{\"id\":\"test.disabled\",\"enabled\":false}]}}")
check_exit(1 --device test.disabled)
file(WRITE "${configuration}" "private malformed configuration")
check_exit(1 --device test.check)
