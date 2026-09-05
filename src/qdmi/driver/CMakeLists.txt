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
    ${TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.."
                              RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/..")
  target_compile_definitions(${TARGET_NAME} PRIVATE QDMI_driver_EXPORTS)

  # Add sources to target
  target_sources(${TARGET_NAME} PRIVATE DeviceRegistry.cpp Driver.cpp)

  # Add headers using file sets
  target_sources(
    ${TARGET_NAME}
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${MQT_CORE_INCLUDE_BUILD_DIR}
           FILES
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/driver/Driver.hpp
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/driver/SessionConfig.hpp)

  # Add link libraries
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi MQT::CoreQDMICommon
    PRIVATE qdmi::qdmi_project_warnings $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
            ${CMAKE_DL_LIBS})

  mqt_get_qdmi_device_targets(QDMI_DEVICE_TARGETS)

  # Ensure the driver can find the device libraries at runtime
  if(QDMI_DEVICE_TARGETS)
    mqt_copy_qdmi_runtime(${TARGET_NAME} ${QDMI_DEVICE_TARGETS})
  endif()

  # add to list of MQT core targets
  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
