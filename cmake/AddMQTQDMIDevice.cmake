# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

include(GNUInstallDirs)

function(_mqt_qdmi_json_escape result value)
  string(REPLACE "\\" "\\\\" escaped "${value}")
  string(REPLACE "\"" "\\\"" escaped "${escaped}")
  string(REPLACE "\n" "\\n" escaped "${escaped}")
  string(REPLACE "\r" "\\r" escaped "${escaped}")
  string(REPLACE "\t" "\\t" escaped "${escaped}")
  set(${result}
      "${escaped}"
      PARENT_SCOPE)
endfunction()

# Configure and register a relocatable built-in QDMI device. The generated fragment is emitted
# beside the runtime library in both build and install trees.
function(mqt_configure_qdmi_device target)
  cmake_parse_arguments(ARG "" "ID;PREFIX" "" ${ARGN})
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Unknown QDMI device target: ${target}")
  endif()
  if(NOT ARG_ID OR NOT ARG_PREFIX)
    message(FATAL_ERROR "mqt_configure_qdmi_device requires ID and PREFIX")
  endif()

  set_target_properties(
    ${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
                         RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
  target_compile_definitions(${target} PRIVATE QDMI_VERSION="${QDMI_VERSION}"
                                               ${ARG_PREFIX}_QDMI_device_EXPORTS)
  _mqt_qdmi_json_escape(device_id "${ARG_ID}")
  _mqt_qdmi_json_escape(device_prefix "${ARG_PREFIX}")

  set(fragment "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${target}.qdmi.json")
  file(
    GENERATE
    OUTPUT "${fragment}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\n        \"id\": \"${device_id}\",\n        \"library\": \"$<TARGET_FILE_NAME:${target}>\",\n        \"prefix\": \"${device_prefix}\",\n        \"enabled\": true\n      }\n    ]\n  }\n}\n"
  )

  add_custom_command(
    TARGET ${target}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${fragment}"
            "$<TARGET_FILE_DIR:${target}>/${target}.qdmi.json")
  set_target_properties(
    ${target}
    PROPERTIES MQT_QDMI_DEVICE_ID "${ARG_ID}"
               MQT_QDMI_DEVICE_PREFIX "${ARG_PREFIX}"
               MQT_QDMI_MANIFEST_NAME "${target}.qdmi.json")
  set_property(GLOBAL APPEND PROPERTY MQT_QDMI_DEVICE_TARGETS ${target})
  set_property(
    TARGET ${target}
    APPEND
    PROPERTY EXPORT_PROPERTIES MQT_QDMI_DEVICE_ID MQT_QDMI_DEVICE_PREFIX MQT_QDMI_MANIFEST_NAME)
  if(WIN32)
    # Shared-library targets are runtime artifacts on Windows and are installed under bin. Keep the
    # fragment beside the DLL so its relative path resolves.
    set(fragment_install_dir ${CMAKE_INSTALL_BINDIR})
  else()
    set(fragment_install_dir ${CMAKE_INSTALL_LIBDIR})
  endif()
  set(install_arguments)
  if(MQT_CORE_TARGET_NAME)
    list(APPEND install_arguments COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
  endif()
  install(
    FILES "${fragment}"
    DESTINATION ${fragment_install_dir}
    ${install_arguments})
endfunction()

# Return every QDMI device registered through mqt_configure_qdmi_device.
function(mqt_get_qdmi_device_targets result)
  get_property(devices GLOBAL PROPERTY MQT_QDMI_DEVICE_TARGETS)
  set(${result}
      ${devices}
      PARENT_SCOPE)
endfunction()

# Copy QDMI device libraries and their manifests beside a static consumer executable.
function(mqt_copy_qdmi_runtime target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Unknown QDMI runtime consumer target: ${target}")
  endif()
  set(devices ${ARGN})
  if(NOT devices)
    mqt_get_qdmi_device_targets(devices)
  endif()
  if(NOT devices)
    message(FATAL_ERROR "mqt_copy_qdmi_runtime requires at least one QDMI device target")
  endif()
  foreach(device IN LISTS devices)
    if(NOT TARGET ${device})
      message(FATAL_ERROR "Unknown QDMI device target: ${device}")
    endif()
    get_target_property(device_target ${device} ALIASED_TARGET)
    if(NOT device_target)
      set(device_target ${device})
    endif()
    get_target_property(manifest_name ${device_target} MQT_QDMI_MANIFEST_NAME)
    if(NOT manifest_name)
      get_target_property(device_id ${device_target} MQT_QDMI_DEVICE_ID)
      get_target_property(device_prefix ${device_target} MQT_QDMI_DEVICE_PREFIX)
      if(NOT device_id OR NOT device_prefix)
        message(
          FATAL_ERROR
            "QDMI device target '${device}' must define either MQT_QDMI_MANIFEST_NAME or both MQT_QDMI_DEVICE_ID and MQT_QDMI_DEVICE_PREFIX"
        )
      endif()
      _mqt_qdmi_json_escape(device_id "${device_id}")
      _mqt_qdmi_json_escape(device_prefix "${device_prefix}")
      string(MAKE_C_IDENTIFIER "${target}-${device}" manifest_stem)
      set(manifest_name "${manifest_stem}.qdmi.json")
      set(manifest "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${manifest_name}")
      file(
        GENERATE
        OUTPUT "${manifest}"
        CONTENT
          "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\n        \"id\": \"${device_id}\",\n        \"library\": \"$<TARGET_FILE_NAME:${device}>\",\n        \"prefix\": \"${device_prefix}\",\n        \"enabled\": true\n      }\n    ]\n  }\n}\n"
      )
    else()
      set(manifest "$<TARGET_FILE_DIR:${device}>/${manifest_name}")
    endif()
    get_target_property(device_imported ${device_target} IMPORTED)
    if(NOT device_imported)
      add_dependencies(${target} ${device})
    endif()
    add_custom_command(
      TARGET ${target}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${device}>"
              "$<TARGET_FILE_DIR:${target}>"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${manifest}"
              "$<TARGET_FILE_DIR:${target}>/${manifest_name}")
  endforeach()
endfunction()
