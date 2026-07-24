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
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME QDMIDriver)

  # Add sources to target
  target_sources(${TARGET_NAME} PRIVATE DeviceRegistry.cpp Driver.cpp)

  # Add headers using file sets
  target_sources(${TARGET_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                       FILES ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/driver/Driver.hpp)

  # Add link libraries
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi MQT::CoreQDMICommon
    PRIVATE qdmi::qdmi_project_warnings nlohmann_json::nlohmann_json spdlog::spdlog
            ${CMAKE_DL_LIBS})
  target_include_directories(${TARGET_NAME} SYSTEM
                             PRIVATE ${PROJECT_SOURCE_DIR}/vendor/tomlplusplus)

  mqt_get_qdmi_device_targets(QDMI_DEVICE_TARGETS)

  # Ensure the driver can find the device libraries at runtime
  if(WIN32)
    mqt_copy_qdmi_runtime(${TARGET_NAME} ${QDMI_DEVICE_TARGETS})
  else()
    add_dependencies(${TARGET_NAME} ${QDMI_DEVICE_TARGETS})
    foreach(device IN LISTS QDMI_DEVICE_TARGETS)
      target_link_options(${TARGET_NAME} INTERFACE
                          $<BUILD_INTERFACE:-Wl,-rpath,$<TARGET_FILE_DIR:${device}>>)
    endforeach()
  endif()

  # add to list of MQT core targets
  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
