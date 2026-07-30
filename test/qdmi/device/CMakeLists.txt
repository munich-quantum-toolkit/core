# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-qdmi-object-model-test)

if(TARGET MQT::CoreQDMI)
  generate_prefixed_qdmi_headers(TEST_LIFECYCLE)
  add_library(mqt-core-qdmi-lifecycle-device SHARED lifecycle_device.cpp)
  target_compile_definitions(mqt-core-qdmi-lifecycle-device
                             PRIVATE TEST_LIFECYCLE_QDMI_device_EXPORTS)
  target_compile_features(mqt-core-qdmi-lifecycle-device PRIVATE cxx_std_20)
  target_include_directories(mqt-core-qdmi-lifecycle-device
                             PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include")
  target_link_libraries(mqt-core-qdmi-lifecycle-device PRIVATE MQT::ProjectOptions
                                                               MQT::ProjectWarnings)

  add_library(mqt-core-qdmi-metadata-device SHARED metadata_device.cpp)
  set_target_properties(
    mqt-core-qdmi-metadata-device
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/device"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/device"
               QDMI_DEVICE_ID "test.metadata-only"
               QDMI_DEVICE_PREFIX "TEST_METADATA")
  set_property(
    TARGET mqt-core-qdmi-metadata-device
    APPEND
    PROPERTY EXPORT_PROPERTIES QDMI_DEVICE_ID QDMI_DEVICE_PREFIX)
  set(metadata_device_export
      "${CMAKE_CURRENT_BINARY_DIR}/mqt-core-qdmi-metadata-device-targets.cmake")
  export(TARGETS mqt-core-qdmi-metadata-device FILE "${metadata_device_export}")

  package_add_test(${TARGET_NAME} MQT::CoreQDMI test_device.cpp test_device_api.cpp)
  target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src/qdmi
                                                    "${CMAKE_CURRENT_BINARY_DIR}/include")
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQDMICommon qdmi::qdmi
                                               mqt-core-qdmi-lifecycle-device)
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE DDSIM_DEVICE_LIBRARY="$<TARGET_FILE:MQT::CoreQDMI_DDSIM_Device>"
            LIFECYCLE_DEVICE_LIBRARY="$<TARGET_FILE:mqt-core-qdmi-lifecycle-device>"
            NA_DEVICE_LIBRARY="$<TARGET_FILE:MQT::CoreQDMINaDevice>"
            SC_DEVICE_LIBRARY="$<TARGET_FILE:MQT::CoreQDMIScDevice>")
  add_dependencies(${TARGET_NAME} MQT::CoreQDMI_DDSIM_Device MQT::CoreQDMINaDevice
                   MQT::CoreQDMIScDevice)
  mqt_copy_qdmi_runtime(${TARGET_NAME})
  mqt_copy_qdmi_runtime(${TARGET_NAME} mqt-core-qdmi-metadata-device)

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
