# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-qdmi-driver-test)

if(TARGET MQT::CoreQDMIDriver)
  add_library(mqt-core-qdmi-metadata-device SHARED metadata_device.cpp)
  set_target_properties(
    mqt-core-qdmi-metadata-device
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/device"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/device"
               QDMI_DEVICE_ID "test.metadata-only"
               QDMI_DEVICE_PREFIX "TEST_METADATA"
               QDMI_RUNTIME_FILES "metadata-runtime.json")
  add_custom_command(
    TARGET mqt-core-qdmi-metadata-device
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/metadata-runtime.json"
      "$<TARGET_FILE_DIR:mqt-core-qdmi-metadata-device>/metadata-runtime.json")
  set_property(
    TARGET mqt-core-qdmi-metadata-device
    APPEND
    PROPERTY EXPORT_PROPERTIES QDMI_DEVICE_ID QDMI_DEVICE_PREFIX QDMI_RUNTIME_FILES)
  set(metadata_device_export
      "${CMAKE_CURRENT_BINARY_DIR}/mqt-core-qdmi-metadata-device-targets.cmake")
  export(TARGETS mqt-core-qdmi-metadata-device FILE "${metadata_device_export}")

  add_library(mqt-core-qdmi-session-device SHARED session_device.cpp)
  target_link_libraries(mqt-core-qdmi-session-device PRIVATE qdmi::qdmi)
  set_target_properties(
    mqt-core-qdmi-session-device
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/device"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/device")

  package_add_test(${TARGET_NAME} MQT::CoreQDMIDriver test_driver.cpp)
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQDMI)
  set(config_file "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/configured-devices.json")
  file(
    GENERATE
    OUTPUT "${config_file}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"mqt.sc.default\", \"library\": \"$<TARGET_FILE:MQT::CoreQDMIScDevice>\", \"prefix\": \"MQT_SC\"},\n      {\"id\": \"mqt.ddsim.default\", \"library\": \"$<TARGET_FILE:MQT::CoreQDMI_DDSIM_Device>\", \"prefix\": \"MQT_DDSIM\"},\n      {\"id\": \"test.disabled\", \"enabled\": false},\n      {\"id\": \"broken.example\", \"library\": \"missing-device-library\", \"prefix\": \"BROKEN\"}\n    ]\n  }\n}\n"
  )
  file(READ "${PROJECT_SOURCE_DIR}/json/sc/mqt-core-qdmi-sc-device.json" custom_sc_json)
  string(REPLACE "MQT SC Default QDMI Device" "Custom SC Driver Device" custom_sc_json
                 "${custom_sc_json}")
  string(REPLACE "\"duration\": 20" "\"duration\": 77" custom_sc_json "${custom_sc_json}")
  set(custom_sc_file "${CMAKE_CURRENT_BINARY_DIR}/custom-sc-device.json")
  file(WRITE "${custom_sc_file}" "${custom_sc_json}")
  string(MAKE_C_IDENTIFIER "${TARGET_NAME}-mqt-core-qdmi-metadata-device" metadata_manifest_stem)
  set(metadata_manifest_file
      "$<TARGET_FILE_DIR:${TARGET_NAME}>/${metadata_manifest_stem}.qdmi.json")
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE
      "MQT_CORE_QDMI_TEST_CONFIG_FILE=\"${config_file}\""
      "MQT_CORE_QDMI_METADATA_MANIFEST=\"${metadata_manifest_file}\""
      "MQT_CORE_QDMI_SESSION_DEVICE=\"$<TARGET_FILE:mqt-core-qdmi-session-device>\""
      "MQT_CORE_QDMI_CUSTOM_SC_FILE=\"${custom_sc_file}\""
      "MQT_CORE_QDMI_DEFAULT_SC_FILE=\"${PROJECT_SOURCE_DIR}/json/sc/mqt-core-qdmi-sc-device.json\""
      "MQT_CORE_QDMI_SC_LIBRARY=\"$<TARGET_FILE:MQT::CoreQDMIScDevice>\""
      "TEST_DEVICE_LIBRARIES=std::array{ std::pair{\"$<TARGET_FILE:MQT::CoreQDMIScDevice>\", \"MQT_SC\"}, std::pair{\"$<TARGET_FILE:MQT::CoreQDMI_DDSIM_Device>\", \"MQT_DDSIM\"} }"
  )

  mqt_copy_qdmi_runtime(${TARGET_NAME})
  mqt_copy_qdmi_runtime(${TARGET_NAME} mqt-core-qdmi-metadata-device)
  add_dependencies(${TARGET_NAME} mqt-core-qdmi-session-device)

  set(imported_device_build_dir "${CMAKE_CURRENT_BINARY_DIR}/imported-device-consumer")
  set(imported_device_configure_command
      ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_SOURCE_DIR}/imported_device" -B
      "${imported_device_build_dir}" -G "${CMAKE_GENERATOR}"
      "-DMQT_CORE_QDMI_DEVICE_TARGETS=${metadata_device_export}"
      "-DMQT_CORE_QDMI_HELPER=${PROJECT_SOURCE_DIR}/cmake/AddMQTQDMIDevice.cmake")
  if(CMAKE_GENERATOR_PLATFORM)
    list(APPEND imported_device_configure_command -A "${CMAKE_GENERATOR_PLATFORM}")
  endif()
  if(CMAKE_GENERATOR_TOOLSET)
    list(APPEND imported_device_configure_command -T "${CMAKE_GENERATOR_TOOLSET}")
  endif()
  add_test(NAME mqt-core-qdmi-imported-device-configure
           COMMAND ${imported_device_configure_command})
  add_test(NAME mqt-core-qdmi-imported-device-build
           COMMAND ${CMAKE_COMMAND} --build "${imported_device_build_dir}" --config $<CONFIG>)
  set_tests_properties(mqt-core-qdmi-imported-device-configure
                       PROPERTIES FIXTURES_SETUP mqt-core-qdmi-imported-device)
  set_tests_properties(mqt-core-qdmi-imported-device-build PROPERTIES FIXTURES_REQUIRED
                                                                      mqt-core-qdmi-imported-device)

  set(runtime_file_device_build_dir "${CMAKE_CURRENT_BINARY_DIR}/runtime-file-device")
  set(runtime_file_device_install_dir "${runtime_file_device_build_dir}/install")
  if(WIN32)
    set(runtime_file_device_artifact_dir "${CMAKE_INSTALL_BINDIR}")
  else()
    set(runtime_file_device_artifact_dir "${CMAKE_INSTALL_LIBDIR}")
  endif()
  set(runtime_file_device_configure_command
      ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_SOURCE_DIR}/runtime_file_device" -B
      "${runtime_file_device_build_dir}" -G "${CMAKE_GENERATOR}"
      "-DMQT_CORE_QDMI_HELPER=${PROJECT_SOURCE_DIR}/cmake/AddMQTQDMIDevice.cmake"
      "-DMQT_CORE_QDMI_TEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}" "-DCMAKE_BUILD_TYPE=$<CONFIG>")
  if(CMAKE_GENERATOR_PLATFORM)
    list(APPEND runtime_file_device_configure_command -A "${CMAKE_GENERATOR_PLATFORM}")
  endif()
  if(CMAKE_GENERATOR_TOOLSET)
    list(APPEND runtime_file_device_configure_command -T "${CMAKE_GENERATOR_TOOLSET}")
  endif()
  add_test(NAME mqt-core-qdmi-runtime-file-configure
           COMMAND ${runtime_file_device_configure_command})
  add_test(NAME mqt-core-qdmi-runtime-file-build
           COMMAND ${CMAKE_COMMAND} --build "${runtime_file_device_build_dir}" --config $<CONFIG>)
  add_test(NAME mqt-core-qdmi-runtime-file-install
           COMMAND ${CMAKE_COMMAND} --install "${runtime_file_device_build_dir}" --config $<CONFIG>
                   --prefix "${runtime_file_device_install_dir}")
  add_test(
    NAME mqt-core-qdmi-runtime-file-install-verify
    COMMAND
      ${CMAKE_COMMAND} -E compare_files
      "${runtime_file_device_install_dir}/${runtime_file_device_artifact_dir}/metadata-runtime.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/metadata-runtime.json")
  set_tests_properties(mqt-core-qdmi-runtime-file-configure
                       PROPERTIES FIXTURES_SETUP mqt-core-qdmi-runtime-file-configure)
  set_tests_properties(
    mqt-core-qdmi-runtime-file-build
    PROPERTIES FIXTURES_REQUIRED mqt-core-qdmi-runtime-file-configure FIXTURES_SETUP
               mqt-core-qdmi-runtime-file-build)
  set_tests_properties(
    mqt-core-qdmi-runtime-file-install
    PROPERTIES FIXTURES_REQUIRED mqt-core-qdmi-runtime-file-build FIXTURES_SETUP
               mqt-core-qdmi-runtime-file-install)
  add_test(
    NAME mqt-core-qdmi-runtime-file-manifest-verify
    COMMAND
      ${CMAKE_COMMAND}
      "-DMANIFEST=${runtime_file_device_install_dir}/${runtime_file_device_artifact_dir}/mqt-core-qdmi-runtime-file-device.qdmi.json"
      "-DEXPECTED=\"id\": \"test.runtime-file.variant\"|\"file\": \"metadata-runtime.json\"|\"id\": \"test.runtime-file.session\"|\"base-url\": \"https://device.example\"|\"custom3\": \"session\""
      -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_generated_manifest.cmake")
  set_tests_properties(mqt-core-qdmi-runtime-file-install-verify
                       PROPERTIES FIXTURES_REQUIRED mqt-core-qdmi-runtime-file-install)
  set_tests_properties(mqt-core-qdmi-runtime-file-manifest-verify
                       PROPERTIES FIXTURES_REQUIRED mqt-core-qdmi-runtime-file-install)

  function(add_qdmi_configuration_rejection_test name invalid_configuration)
    set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/runtime-file-device-invalid-${name}")
    set(configure_command
        ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_SOURCE_DIR}/runtime_file_device" -B "${build_dir}" -G
        "${CMAKE_GENERATOR}"
        "-DMQT_CORE_QDMI_HELPER=${PROJECT_SOURCE_DIR}/cmake/AddMQTQDMIDevice.cmake"
        "-DMQT_CORE_QDMI_TEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DMQT_CORE_QDMI_INVALID_CONFIGURATION=${invalid_configuration}"
        "-DCMAKE_BUILD_TYPE=$<CONFIG>")
    if(CMAKE_GENERATOR_PLATFORM)
      list(APPEND configure_command -A "${CMAKE_GENERATOR_PLATFORM}")
    endif()
    if(CMAKE_GENERATOR_TOOLSET)
      list(APPEND configure_command -T "${CMAKE_GENERATOR_TOOLSET}")
    endif()
    add_test(NAME "mqt-core-qdmi-configuration-rejects-${name}" COMMAND ${configure_command})
    set_tests_properties("mqt-core-qdmi-configuration-rejects-${name}" PROPERTIES WILL_FAIL TRUE)
  endfunction()

  add_qdmi_configuration_rejection_test(empty-id empty-id)
  add_qdmi_configuration_rejection_test(empty-file empty-file)
  add_qdmi_configuration_rejection_test(duplicate-base-id duplicate-base-id)
  add_qdmi_configuration_rejection_test(duplicate-configured-id duplicate-configured-id)
  add_qdmi_configuration_rejection_test(missing-session-parameter missing-session-parameter)
  add_qdmi_configuration_rejection_test(malformed-session-parameter malformed-session-parameter)
  add_qdmi_configuration_rejection_test(empty-session-value empty-session-value)
  add_qdmi_configuration_rejection_test(unknown-session-parameter unknown-session-parameter)
  add_qdmi_configuration_rejection_test(credential-session-parameter credential-session-parameter)
  add_qdmi_configuration_rejection_test(repeated-session-parameter repeated-session-parameter)
  add_qdmi_configuration_rejection_test(unknown-session-runtime-file unknown-session-runtime-file)
  add_qdmi_configuration_rejection_test(reserved-session-parameter reserved-session-parameter)
  add_qdmi_configuration_rejection_test(duplicate-session-id duplicate-session-id)
endif()
