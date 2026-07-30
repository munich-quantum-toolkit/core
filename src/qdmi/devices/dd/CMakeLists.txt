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
  add_mqt_core_library(${TARGET_NAME} FORCE_SHARED HIDDEN_VISIBILITY ALIAS_NAME QDMI_DDSIM_Device)

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
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/devices/dd/Device.hpp
           ${QDMI_HDRS})

  # Add link libraries
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreDD MQT::CoreQASM MQT::CoreCircuitOptimizer
                                               MQT::CoreQDMICommon spdlog::spdlog)
  if(BUILD_MQT_CORE_QDMI_DDSIM_WITH_QIR)
    target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQIRJIT MQT::CoreQIRRuntime)
    target_compile_definitions(${TARGET_NAME} PRIVATE BUILD_MQT_CORE_QDMI_DDSIM_WITH_QIR)
  endif()

  mqt_configure_qdmi_device(${TARGET_NAME} ID mqt.ddsim.default PREFIX ${QDMI_PREFIX})
  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
