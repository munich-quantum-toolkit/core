# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-driver)

if(NOT TARGET ${TARGET_NAME})
  # Add driver library
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME QDMIDriver FORCE_SHARED)
  set_target_properties(
    ${TARGET_NAME}
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.."
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.."
               WINDOWS_EXPORT_ALL_SYMBOLS OFF)
  target_compile_definitions(${TARGET_NAME} PRIVATE QDMI_driver_EXPORTS)

  target_sources(${TARGET_NAME} PRIVATE DeviceRegistry.cpp Driver.cpp)

  target_sources(
    ${TARGET_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR} FILES
                          ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/driver/DriverExtension.hpp)

  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi MQT::CoreQDMICommon
    PRIVATE qdmi::qdmi_project_warnings $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
            ${CMAKE_DL_LIBS})

  configure_qdmi_exports(
    TARGET
    ${TARGET_NAME}
    INTERFACE
    client
    EXTRA_SYMBOLS
    MQT_CORE_QDMI_driver_add_manifest_v1
    MQT_CORE_QDMI_driver_registered_device_ids_v1
    MQT_CORE_QDMI_driver_session_alloc_for_device_v1)

  if(BUILD_MQT_CORE_TESTS)
    add_library(mqt-core-qdmi-driver-testing STATIC DeviceRegistry.cpp Driver.cpp)
    target_compile_features(mqt-core-qdmi-driver-testing PUBLIC cxx_std_20)
    target_compile_definitions(mqt-core-qdmi-driver-testing PUBLIC QDMI_STATIC_DEFINE)
    target_link_libraries(
      mqt-core-qdmi-driver-testing
      PUBLIC qdmi::qdmi MQT::CoreQDMICommon
      PRIVATE MQT::ProjectOptions qdmi::qdmi_project_warnings nlohmann_json::nlohmann_json
              ${CMAKE_DL_LIBS})
  endif()

  mqt_get_qdmi_device_targets(QDMI_DEVICE_TARGETS)

  # Ensure the driver can find the device libraries at runtime
  if(QDMI_DEVICE_TARGETS)
    mqt_copy_qdmi_runtime(${TARGET_NAME} ${QDMI_DEVICE_TARGETS})
  endif()

  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
