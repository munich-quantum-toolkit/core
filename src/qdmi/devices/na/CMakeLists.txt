# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Set target name
set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-na-device-config)

# If the target is not already defined
if(NOT TARGET ${TARGET_NAME})

  # Add the runtime configuration parser library
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME QDMINaDeviceConfig)

  # add sources to target
  target_sources(${TARGET_NAME} PRIVATE Configuration.cpp)

  # add headers using file sets
  target_sources(
    ${TARGET_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR} FILES
                          ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/devices/na/Configuration.hpp)

  target_link_libraries(${TARGET_NAME} PUBLIC nlohmann_json::nlohmann_json)

  # add to list of MQT core targets
  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})

  # Place the library in the common output directories.
  set_target_properties(
    ${TARGET_NAME}
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
endif()

if(NOT BUILD_MQT_CORE_QDMI_NA_DEVICE)
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS}
      PARENT_SCOPE)
  return()
endif()

# Set target name
set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-na-device)

# Set prefix for QDMI
set(QDMI_PREFIX "MQT_NA")

# If the target is not already defined
if(NOT TARGET ${TARGET_NAME})

  # Add library
  add_mqt_core_library(${TARGET_NAME} FORCE_SHARED HIDDEN_VISIBILITY ALIAS_NAME QDMINaDevice)

  # Generate prefixed QDMI headers
  generate_prefixed_qdmi_headers(${QDMI_PREFIX})
  file(GLOB_RECURSE QDMI_HDRS ${CMAKE_CURRENT_BINARY_DIR}/include/mqt_na_qdmi/**.h)

  # add sources to target
  target_sources(${TARGET_NAME} PRIVATE Device.cpp)

  # add headers using file sets
  target_sources(
    ${TARGET_NAME}
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${MQT_CORE_INCLUDE_BUILD_DIR}
           ${CMAKE_CURRENT_BINARY_DIR}/include
           FILES
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/devices/na/Device.hpp
           ${QDMI_HDRS})

  # add link libraries
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQDMICommon MQT::CoreQDMINaDeviceConfig
                                               spdlog::spdlog)

  mqt_configure_qdmi_device(
    ${TARGET_NAME}
    ID
    mqt.na.default
    PREFIX
    ${QDMI_PREFIX}
    RUNTIME_FILES
    ${PROJECT_SOURCE_DIR}/json/na/mqt-core-qdmi-na-device.json)
  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
