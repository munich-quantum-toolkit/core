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

  # Share the implementation with lifecycle tests without exporting C++ internals.
  add_library(${TARGET_NAME}-objects OBJECT Device.cpp)
  set_target_properties(
    ${TARGET_NAME}-objects
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON)
  target_include_directories(${TARGET_NAME}-objects PRIVATE ${MQT_CORE_INCLUDE_BUILD_DIR}
                                                            ${CMAKE_CURRENT_BINARY_DIR}/include)
  target_compile_definitions(${TARGET_NAME}-objects
                             PRIVATE $<TARGET_PROPERTY:${TARGET_NAME},COMPILE_DEFINITIONS>)
  target_link_libraries(${TARGET_NAME} PRIVATE ${TARGET_NAME}-objects)

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
  target_link_libraries(${TARGET_NAME}-objects PUBLIC MQT::CoreDD MQT::CoreQDMICommon
                                                      MQT::ProjectWarnings MQT::ProjectOptions)
  mqt_llvm_target_disable_rtti(${TARGET_NAME}-objects)
  if(UNIX AND NOT APPLE)
    # Keep symbols from static dependencies local. A plugin-enabled host can otherwise interpose
    # LLVM's process-wide registries when it loads this device.
    target_link_options(${TARGET_NAME} PRIVATE "LINKER:--exclude-libs,ALL")
  endif()
  target_link_libraries(${TARGET_NAME}-objects PUBLIC MLIRQCODDFunctionality MQTCompilerPrograms
                                                      MQT::CoreQIRJIT MQT::CoreQIRRuntime)

  mqt_configure_qdmi_device(${TARGET_NAME} ID mqt.ddsim.default PREFIX ${QDMI_PREFIX})
  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
