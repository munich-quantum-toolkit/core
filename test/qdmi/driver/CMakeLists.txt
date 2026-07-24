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
               MQT_QDMI_DEVICE_ID "test.metadata-only"
               MQT_QDMI_DEVICE_PREFIX "TEST_METADATA")
  set_property(
    TARGET mqt-core-qdmi-metadata-device
    APPEND
    PROPERTY EXPORT_PROPERTIES MQT_QDMI_DEVICE_ID MQT_QDMI_DEVICE_PREFIX)
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
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreFoMaC)
  set(config_file "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/configured-devices.json")
  file(
    GENERATE
    OUTPUT "${config_file}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"mqt.na.default\", \"library\": \"$<TARGET_FILE:MQT::CoreQDMINaDevice>\", \"prefix\": \"MQT_NA\"},\n      {\"id\": \"mqt.sc.default\", \"library\": \"$<TARGET_FILE:MQT::CoreQDMIScDevice>\", \"prefix\": \"MQT_SC\"},\n      {\"id\": \"mqt.ddsim.default\", \"library\": \"$<TARGET_FILE:MQT::CoreQDMI_DDSIM_Device>\", \"prefix\": \"MQT_DDSIM\"},\n      {\"id\": \"test.disabled\", \"enabled\": false},\n      {\"id\": \"broken.example\", \"library\": \"missing-device-library\", \"prefix\": \"BROKEN\"}\n    ]\n  }\n}\n"
  )
  string(MAKE_C_IDENTIFIER "${TARGET_NAME}-mqt-core-qdmi-metadata-device" metadata_manifest_stem)
  set(metadata_manifest_file
      "$<TARGET_FILE_DIR:${TARGET_NAME}>/${metadata_manifest_stem}.qdmi.json")
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE
      "MQT_CORE_QDMI_TEST_CONFIG_FILE=\"${config_file}\""
      "MQT_CORE_QDMI_METADATA_MANIFEST=\"${metadata_manifest_file}\""
      "MQT_CORE_QDMI_SESSION_DEVICE=\"$<TARGET_FILE:mqt-core-qdmi-session-device>\""
      "TEST_DEVICE_LIBRARIES=std::array{ std::pair{\"$<TARGET_FILE:MQT::CoreQDMINaDevice>\", \"MQT_NA\"}, std::pair{\"$<TARGET_FILE:MQT::CoreQDMIScDevice>\", \"MQT_SC\"}, std::pair{\"$<TARGET_FILE:MQT::CoreQDMI_DDSIM_Device>\", \"MQT_DDSIM\"} }"
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
endif()
