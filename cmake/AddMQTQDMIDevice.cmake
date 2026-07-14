# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

include(GNUInstallDirs)

# Register a relocatable built-in QDMI device. The generated fragment is emitted beside the runtime
# library in both build and install trees.
function(mqt_register_qdmi_device target)
  cmake_parse_arguments(ARG "" "ID;PREFIX" "SESSION" ${ARGN})
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Unknown QDMI device target: ${target}")
  endif()
  if(NOT ARG_ID OR NOT ARG_PREFIX)
    message(FATAL_ERROR "mqt_register_qdmi_device requires ID and PREFIX")
  endif()

  set(session_json "")
  foreach(pair IN LISTS ARG_SESSION)
    string(REPLACE "=" ";" parts "${pair}")
    list(LENGTH parts count)
    if(NOT count EQUAL 2)
      message(FATAL_ERROR "Invalid QDMI session default '${pair}'")
    endif()
    list(GET parts 0 key)
    list(GET parts 1 value)
    string(APPEND session_json "\n          \"${key}\": \"${value}\",")
  endforeach()
  string(REGEX REPLACE ",$" "" session_json "${session_json}")

  set(fragment "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${target}.qdmi.json")
  file(
    GENERATE
    OUTPUT "${fragment}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\n        \"id\": \"${ARG_ID}\",\n        \"library\": \"$<TARGET_FILE_NAME:${target}>\",\n        \"abi\": \"qdmi-v1\",\n        \"prefix\": \"${ARG_PREFIX}\",\n        \"enabled\": true,\n        \"session\": {${session_json}\n        }\n      }\n    ]\n  }\n}\n"
  )

  add_custom_command(
    TARGET ${target}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${fragment}"
            "$<TARGET_FILE_DIR:${target}>/${target}.qdmi.json")
  install(
    FILES "${fragment}"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
endfunction()
