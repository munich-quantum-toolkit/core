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
  cmake_parse_arguments(ARG "" "ID;PREFIX" "RUNTIME_FILES;CONFIGURATIONS" ${ARGN})
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

  set(device_entries
      "      {\n        \"id\": \"${device_id}\",\n        \"library\": \"$<TARGET_FILE_NAME:${target}>\",\n        \"prefix\": \"${device_prefix}\",\n        \"enabled\": true\n      }"
  )
  set(configuration_ids "${ARG_ID}")
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
    list(FIND configuration_ids "${configuration_id}" configuration_id_index)
    if(NOT configuration_id_index EQUAL -1)
      message(FATAL_ERROR "Duplicate QDMI device ID '${configuration_id}'")
    endif()
    list(APPEND configuration_ids "${configuration_id}")
    list(FIND runtime_file_names "${configuration_file}" runtime_file_index)
    if(runtime_file_index EQUAL -1)
      message(
        FATAL_ERROR
          "QDMI configuration '${configuration_id}' refers to unknown runtime file '${configuration_file}'"
      )
    endif()
    _mqt_qdmi_json_escape(configuration_id "${configuration_id}")
    _mqt_qdmi_json_escape(configuration_file "${configuration_file}")
    string(
      APPEND
      device_entries
      ",\n      {\n        \"id\": \"${configuration_id}\",\n        \"library\": \"$<TARGET_FILE_NAME:${target}>\",\n        \"prefix\": \"${device_prefix}\",\n        \"enabled\": true,\n        \"session\": {\n          \"device-config\": {\n            \"file\": \"${configuration_file}\"\n          }\n        }\n      }"
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

# Copy the shared QDMI libraries, device manifests, and assets beside an application.
function(mqt_copy_qdmi_runtime target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Unknown QDMI application target: ${target}")
  endif()
  set_property(TARGET ${target} PROPERTY BUILD_WITH_INSTALL_RPATH FALSE)
  set(devices ${ARGN})
  if(NOT devices)
    mqt_get_qdmi_device_targets(devices)
  endif()
  set(libraries ${devices})
  foreach(library IN ITEMS MQT::CoreQDMI MQT::CoreQDMIDriver)
    if(TARGET ${library})
      list(APPEND libraries ${library})
    endif()
  endforeach()
  if(NOT libraries)
    message(FATAL_ERROR "mqt_copy_qdmi_runtime requires a QDMI driver or device target")
  endif()
  list(REMOVE_DUPLICATES libraries)
  foreach(library IN LISTS libraries)
    if(NOT TARGET ${library})
      message(FATAL_ERROR "Unknown QDMI library target: ${library}")
    endif()
    get_target_property(library_target ${library} ALIASED_TARGET)
    if(NOT library_target)
      set(library_target ${library})
    endif()
    get_target_property(library_type ${library_target} TYPE)
    if(library_target STREQUAL "${target}" OR NOT library_type MATCHES "^(SHARED|MODULE)_LIBRARY$")
      continue()
    endif()
    get_target_property(imported ${library_target} IMPORTED)
    set(files "$<TARGET_FILE:${library}>")
    if(WIN32)
      if(imported)
        # TARGET_RUNTIME_DLLS needs a local target to traverse imported dependencies.
        string(MAKE_C_IDENTIFIER "${library_target}-dependencies" dependency_target)
        if(NOT TARGET ${dependency_target})
          add_library(${dependency_target} MODULE EXCLUDE_FROM_ALL
                      "${CMAKE_CURRENT_FUNCTION_LIST_FILE}")
          set_property(TARGET ${dependency_target} PROPERTY LINKER_LANGUAGE CXX)
          target_link_libraries(${dependency_target} PRIVATE ${library})
        endif()
        set(files "$<TARGET_RUNTIME_DLLS:${dependency_target}>")
      else()
        list(APPEND files "$<TARGET_RUNTIME_DLLS:${library}>")
      endif()
    endif()
    if(library IN_LIST devices)
      get_target_property(manifest_name ${library_target} QDMI_MANIFEST_NAME)
      if(NOT manifest_name)
        get_target_property(device_id ${library_target} QDMI_DEVICE_ID)
        get_target_property(device_prefix ${library_target} QDMI_DEVICE_PREFIX)
        if(NOT device_id OR NOT device_prefix)
          message(
            FATAL_ERROR
              "QDMI device target '${library}' must define either QDMI_MANIFEST_NAME or both QDMI_DEVICE_ID and QDMI_DEVICE_PREFIX"
          )
        endif()
        _mqt_qdmi_json_escape(device_id "${device_id}")
        _mqt_qdmi_json_escape(device_prefix "${device_prefix}")
        string(MAKE_C_IDENTIFIER "${target}-${library}" manifest_stem)
        set(manifest_name "${manifest_stem}.qdmi.json")
        set(manifest "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${manifest_name}")
        file(
          GENERATE
          OUTPUT "${manifest}"
          CONTENT
            "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\n        \"id\": \"${device_id}\",\n        \"library\": \"$<TARGET_FILE_NAME:${library}>\",\n        \"prefix\": \"${device_prefix}\",\n        \"enabled\": true\n      }\n    ]\n  }\n}\n"
        )
      else()
        set(manifest "$<TARGET_FILE_DIR:${library}>/${manifest_name}")
      endif()
      list(APPEND files "${manifest}")
      get_target_property(assets ${library_target} QDMI_RUNTIME_FILES)
      if(assets)
        foreach(asset IN LISTS assets)
          list(APPEND files "$<TARGET_FILE_DIR:${library}>/${asset}")
        endforeach()
      endif()
    endif()
    if(NOT imported)
      add_dependencies(${target} ${library_target})
    endif()
    add_custom_command(
      TARGET ${target}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} "$<TARGET_FILE_DIR:${target}>"
      COMMAND_EXPAND_LISTS)
    if(NOT WIN32 AND imported)
      add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} "-DLIBRARY=$<TARGET_FILE:${library}>"
          "-DDESTINATION=$<TARGET_FILE_DIR:${target}>" -P
          "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CopyQDMISharedDependencies.cmake")
    endif()
  endforeach()
endfunction()
