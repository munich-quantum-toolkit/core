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

# Build the JSON session object of a generated QDMI device entry. Every argument after
# runtime_file_names is one '<key>=<value>' session parameter.
function(_mqt_qdmi_session_object result device_id runtime_file_names)
  # A generated fragment is installed with the package and is world-readable, so credentials and
  # host-specific values belong in a trusted registry file such as /etc/mqt-core/qdmi.json.
  set(credential_keys token password auth-file username)
  set(allowed_keys
      base-url
      auth-url
      custom1
      custom2
      custom3
      custom4
      custom5
      device-config-file)
  set(members)
  set(seen_keys)
  set(configuration_file)
  foreach(parameter IN LISTS ARGN)
    string(FIND "${parameter}" "=" separator)
    if(separator EQUAL -1)
      message(
        FATAL_ERROR
          "QDMI device '${device_id}' declares '${parameter}', but a session parameter must use the form '<key>=<value>'"
      )
    endif()
    string(SUBSTRING "${parameter}" 0 ${separator} key)
    math(EXPR value_begin "${separator} + 1")
    string(SUBSTRING "${parameter}" ${value_begin} -1 value)
    if(key STREQUAL "" OR value STREQUAL "")
      message(
        FATAL_ERROR
          "QDMI device '${device_id}' declares '${parameter}', but a session parameter must have a non-empty key and value"
      )
    endif()
    list(FIND credential_keys "${key}" credential_index)
    if(NOT credential_index EQUAL -1)
      message(
        FATAL_ERROR
          "QDMI device '${device_id}' must not declare the session parameter '${key}' in a generated fragment; declare it in a trusted registry file such as /etc/mqt-core/qdmi.json"
      )
    endif()
    list(FIND allowed_keys "${key}" key_index)
    if(key_index EQUAL -1)
      string(REPLACE ";" ", " allowed_text "${allowed_keys}")
      message(
        FATAL_ERROR
          "QDMI device '${device_id}' uses the unknown session parameter '${key}'; the supported parameters are ${allowed_text}"
      )
    endif()
    list(FIND seen_keys "${key}" seen_index)
    if(NOT seen_index EQUAL -1)
      message(FATAL_ERROR "QDMI device '${device_id}' repeats the session parameter '${key}'")
    endif()
    list(APPEND seen_keys "${key}")
    if(key STREQUAL "device-config-file")
      list(FIND runtime_file_names "${value}" runtime_file_index)
      if(runtime_file_index EQUAL -1)
        message(
          FATAL_ERROR "QDMI device '${device_id}' refers to the unknown runtime file '${value}'")
      endif()
      set(configuration_file "${value}")
    else()
      _mqt_qdmi_json_escape(escaped_value "${value}")
      list(APPEND members "          \"${key}\": \"${escaped_value}\"")
    endif()
  endforeach()

  if(configuration_file)
    # The Driver adapts a device configuration file to QDMI CUSTOM2 and inline JSON to CUSTOM1.
    foreach(reserved IN ITEMS custom1 custom2)
      list(FIND seen_keys "${reserved}" reserved_index)
      if(NOT reserved_index EQUAL -1)
        message(
          FATAL_ERROR
            "QDMI device '${device_id}' combines 'device-config-file' with the reserved session parameter '${reserved}'"
        )
      endif()
    endforeach()
    _mqt_qdmi_json_escape(escaped_file "${configuration_file}")
    list(APPEND members
         "          \"device-config\": {\n            \"file\": \"${escaped_file}\"\n          }")
  endif()

  list(JOIN members ",\n" joined_members)
  set(${result}
      "{\n${joined_members}\n        }"
      PARENT_SCOPE)
endfunction()

# Configure and register a relocatable built-in QDMI device. The generated fragment is emitted
# beside the runtime library in both build and install trees.
#
# CONFIGURATIONS adds one device entry per '<device-id>|<runtime-file-name>' element, while DEVICES
# adds one entry per '<device-id>|<key>=<value>' element and accepts several parameters at once, for
# example 'example.qc.alpha|base-url=https://alpha.example|custom2=alpha'. A parameter value must
# not contain '|', ';', or whitespace.
function(mqt_configure_qdmi_device target)
  cmake_parse_arguments(ARG "" "ID;PREFIX" "RUNTIME_FILES;CONFIGURATIONS;DEVICES" ${ARGN})
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

  set(runtime_file_names)
  foreach(runtime_file IN LISTS ARG_RUNTIME_FILES)
    get_filename_component(runtime_file_name "${runtime_file}" NAME)
    list(APPEND runtime_file_names "${runtime_file_name}")
    add_custom_command(
      TARGET ${target}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${runtime_file}"
              "$<TARGET_FILE_DIR:${target}>/${runtime_file_name}")
  endforeach()

  # Rewrite the CONFIGURATIONS shorthand so that one loop emits every generated entry.
  set(device_specifications)
  foreach(configuration IN LISTS ARG_CONFIGURATIONS)
    string(REPLACE "|" ";" configuration_parts "${configuration}")
    list(LENGTH configuration_parts configuration_length)
    if(NOT configuration_length EQUAL 2)
      message(
        FATAL_ERROR
          "QDMI configuration '${configuration}' must use the form '<device-id>|<runtime-file-name>'"
      )
    endif()
    list(GET configuration_parts 0 configuration_id)
    list(GET configuration_parts 1 configuration_file)
    if(configuration_id STREQUAL "" OR configuration_file STREQUAL "")
      message(
        FATAL_ERROR
          "QDMI configuration '${configuration}' must use a non-empty device ID and runtime file name"
      )
    endif()
    list(APPEND device_specifications
         "${configuration_id}|device-config-file=${configuration_file}")
  endforeach()
  list(APPEND device_specifications ${ARG_DEVICES})

  set(device_entries
      "      {\n        \"id\": \"${device_id}\",\n        \"library\": \"$<TARGET_FILE_NAME:${target}>\",\n        \"prefix\": \"${device_prefix}\",\n        \"enabled\": true\n      }"
  )
  set(device_ids "${ARG_ID}")
  foreach(specification IN LISTS device_specifications)
    string(REPLACE "|" ";" specification_parts "${specification}")
    list(LENGTH specification_parts specification_length)
    if(specification_length LESS 2)
      message(
        FATAL_ERROR
          "QDMI device '${specification}' must use the form '<device-id>|<key>=<value>' and declare at least one session parameter"
      )
    endif()
    list(POP_FRONT specification_parts specification_id)
    if(specification_id STREQUAL "")
      message(FATAL_ERROR "QDMI device '${specification}' must use a non-empty device ID")
    endif()
    list(FIND device_ids "${specification_id}" specification_id_index)
    if(NOT specification_id_index EQUAL -1)
      message(FATAL_ERROR "Duplicate QDMI device ID '${specification_id}'")
    endif()
    list(APPEND device_ids "${specification_id}")
    _mqt_qdmi_session_object(session_object "${specification_id}" "${runtime_file_names}"
                             ${specification_parts})
    _mqt_qdmi_json_escape(escaped_specification_id "${specification_id}")
    string(
      APPEND
      device_entries
      ",\n      {\n        \"id\": \"${escaped_specification_id}\",\n        \"library\": \"$<TARGET_FILE_NAME:${target}>\",\n        \"prefix\": \"${device_prefix}\",\n        \"enabled\": true,\n        \"session\": ${session_object}\n      }"
    )
  endforeach()

  set(fragment "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${target}.qdmi.json")
  file(
    GENERATE
    OUTPUT "${fragment}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n${device_entries}\n    ]\n  }\n}\n"
  )

  add_custom_command(
    TARGET ${target}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${fragment}"
            "$<TARGET_FILE_DIR:${target}>/${target}.qdmi.json")
  set_target_properties(
    ${target}
    PROPERTIES QDMI_DEVICE_ID "${ARG_ID}"
               QDMI_DEVICE_PREFIX "${ARG_PREFIX}"
               QDMI_MANIFEST_NAME "${target}.qdmi.json"
               QDMI_RUNTIME_FILES "${runtime_file_names}")
  set_property(GLOBAL APPEND PROPERTY MQT_QDMI_DEVICE_TARGETS ${target})
  set_property(
    TARGET ${target}
    APPEND
    PROPERTY EXPORT_PROPERTIES QDMI_DEVICE_ID QDMI_DEVICE_PREFIX QDMI_MANIFEST_NAME
             QDMI_RUNTIME_FILES)
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
  if(ARG_RUNTIME_FILES)
    install(
      FILES ${ARG_RUNTIME_FILES}
      DESTINATION ${fragment_install_dir}
      ${install_arguments})
  endif()
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
    get_target_property(manifest_name ${device_target} QDMI_MANIFEST_NAME)
    if(NOT manifest_name)
      get_target_property(device_id ${device_target} QDMI_DEVICE_ID)
      get_target_property(device_prefix ${device_target} QDMI_DEVICE_PREFIX)
      if(NOT device_id OR NOT device_prefix)
        message(
          FATAL_ERROR
            "QDMI device target '${device}' must define either QDMI_MANIFEST_NAME or both QDMI_DEVICE_ID and QDMI_DEVICE_PREFIX"
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
    get_target_property(runtime_files ${device_target} QDMI_RUNTIME_FILES)
    if(runtime_files)
      foreach(runtime_file IN LISTS runtime_files)
        add_custom_command(
          TARGET ${target}
          POST_BUILD
          COMMAND
            ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE_DIR:${device}>/${runtime_file}"
            "$<TARGET_FILE_DIR:${target}>/${runtime_file}")
      endforeach()
    endif()
  endforeach()
endfunction()
