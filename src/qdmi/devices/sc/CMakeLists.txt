# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-sc-device-config)

if(NOT TARGET ${TARGET_NAME})

  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME QDMIScDeviceConfig)

  target_sources(${TARGET_NAME} PRIVATE Configuration.cpp)

  target_sources(
    ${TARGET_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR} FILES
                          ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/devices/sc/Configuration.hpp)

  target_link_libraries(${TARGET_NAME} PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)

  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})

  # Place the library in the common output directories.
  set_target_properties(
    ${TARGET_NAME}
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
endif()

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-sc-device)

set(QDMI_PREFIX "MQT_SC")

if(NOT TARGET ${TARGET_NAME})

  add_mqt_core_library(${TARGET_NAME} FORCE_SHARED HIDDEN_VISIBILITY ALIAS_NAME QDMIScDevice)

  generate_prefixed_qdmi_headers(${QDMI_PREFIX})
  file(GLOB_RECURSE QDMI_HDRS ${CMAKE_CURRENT_BINARY_DIR}/include/mqt_sc_qdmi/**.h)

  target_sources(${TARGET_NAME} PRIVATE Device.cpp)

  target_sources(
    ${TARGET_NAME}
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${MQT_CORE_INCLUDE_BUILD_DIR}
           ${CMAKE_CURRENT_BINARY_DIR}/include
           FILES
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/devices/sc/Device.hpp
           ${QDMI_HDRS})

  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQDMICommon MQT::CoreQDMIScDeviceConfig)

  mqt_configure_qdmi_device(
    ${TARGET_NAME}
    ID
    mqt.sc.default
    PREFIX
    ${QDMI_PREFIX}
    RUNTIME_FILES
    ${PROJECT_SOURCE_DIR}/json/sc/mqt-core-qdmi-sc-device.json
    ${PROJECT_SOURCE_DIR}/json/sc/iqm-garnet.json
    ${PROJECT_SOURCE_DIR}/json/sc/iqm-emerald.json
    CONFIGURATIONS
    "mqt.sc.iqm.garnet|iqm-garnet.json"
    "mqt.sc.iqm.emerald|iqm-emerald.json")
  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
