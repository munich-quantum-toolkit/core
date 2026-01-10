# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Set target name
set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-ddsim-device)

# Set prefix for QDMI
set(QDMI_PREFIX "MQT_DDSIM")

# If the target is not already defined
if(NOT TARGET ${TARGET_NAME})

  # Add library
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME QDMI_DDSIM_Device)

  # Generate prefixed QDMI headers
  generate_prefixed_qdmi_headers(${QDMI_PREFIX})
  file(GLOB_RECURSE QDMI_HDRS ${CMAKE_CURRENT_BINARY_DIR}/include/mqt_ddsim_qdmi/**.h)

  # Add sources to target
  target_sources(${TARGET_NAME} PRIVATE Device.cpp)

  # Add headers using file sets
  target_sources(
    ${TARGET_NAME}
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${MQT_CORE_INCLUDE_BUILD_DIR}
           ${CMAKE_CURRENT_BINARY_DIR}/include
           FILES
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/dd/Device.hpp
           ${QDMI_HDRS})

  # Add link libraries
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi
    PRIVATE MQT::CoreDD MQT::CoreQASM MQT::CoreCircuitOptimizer MQT::CoreQDMICommon spdlog::spdlog)

  # Always compile with position independent code such that the library can be used in shared
  # libraries
  set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # Add to list of MQT Core targets
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${TARGET_NAME}
      PARENT_SCOPE)

  # Make QDMI version available
  target_compile_definitions(${TARGET_NAME} PRIVATE QDMI_VERSION="${QDMI_VERSION}")

  # Generate additional alias for the target required for generate_device_defs_executable function
  # in the tests
  add_library(qdmi::mqt_ddsim_device ALIAS ${TARGET_NAME})
endif()
