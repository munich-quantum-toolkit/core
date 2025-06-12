# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Set target name
set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-na-device)

# Set prefix for QDMI
set(QDMI_PREFIX "MQT_NA")

# If the target is not already defined
if(NOT TARGET ${TARGET_NAME})

  # Set paths
  set(JSON_FILE ${PROJECT_SOURCE_DIR}/json/na/device.json)
  set(DEVICE_HDR ${CMAKE_CURRENT_BINARY_DIR}/include/Device.hpp)

  # Generate definitions for device
  add_custom_command(
    OUTPUT ${DEVICE_HDR}
    COMMAND MQT::CoreNADeviceGen ARGS --hpp_out=${CMAKE_CURRENT_BINARY_DIR}/include ${JSON_FILE}
    DEPENDS ${JSON_FILE}
    COMMENT "Generating C++ header from ${JSON_FILE}")
  add_custom_target(generate_na_device_header DEPENDS ${DEVICE_HDR})

  # Add library
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME NaQDMI device.cpp)

  # Generate prefixed QDMI headers
  generate_prefixed_qdmi_headers(${QDMI_PREFIX})

  # Add include directories
  target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)

  # Link Protobuf, QDMI, spdlog, and project warnings
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi spdlog::spdlog
    PRIVATE qdmi::project_warnings MQT::ProjectOptions MQT::ProjectWarnings)

  # Make QDMI and MQT Core Version available
  target_compile_definitions(${TARGET_NAME} PRIVATE QDMI_VERSION="${QDMI_VERSION}")
  target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_VERSION="${MQT_CORE_VERSION}")

  # Generate additional alias for the target
  add_library(qdmi::mqt_na_device ALIAS ${TARGET_NAME})
endif()
