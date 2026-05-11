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
  target_sources(${TARGET_NAME} PRIVATE Driver.cpp)

  # Add headers using file sets
  target_sources(${TARGET_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                       FILES ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/driver/Driver.hpp)

  # Add link libraries
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi MQT::CoreQDMICommon
    PRIVATE qdmi::qdmi_project_warnings spdlog::spdlog ${CMAKE_DL_LIBS})

  add_dependencies(${TARGET_NAME} MQT::CoreQDMINaDevice MQT::CoreQDMIScDevice
                   MQT::CoreQDMI_DDSIM_Device)
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE
      "DYN_DEV_LIBS=std::pair{\"$<TARGET_FILE_NAME:MQT::CoreQDMINaDevice>\", \"MQT_NA\"}, std::pair{\"$<TARGET_FILE_NAME:MQT::CoreQDMIScDevice>\", \"MQT_SC\"}, std::pair{\"$<TARGET_FILE_NAME:MQT::CoreQDMI_DDSIM_Device>\", \"MQT_DDSIM\"}"
  )

  # Ensure the driver can find the device libraries at runtime
  if(WIN32)
    # On Windows, we need to copy the device DLLs to the library directory
    add_custom_command(
      TARGET ${TARGET_NAME}
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:MQT::CoreQDMIScDevice>
              $<TARGET_FILE_DIR:${TARGET_NAME}>)
    add_custom_command(
      TARGET ${TARGET_NAME}
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:MQT::CoreQDMINaDevice>
              $<TARGET_FILE_DIR:${TARGET_NAME}>)
    add_custom_command(
      TARGET ${TARGET_NAME}
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:MQT::CoreQDMI_DDSIM_Device>
              $<TARGET_FILE_DIR:${TARGET_NAME}>)
  else()
    target_link_options(
      ${TARGET_NAME}
      INTERFACE
      $<BUILD_INTERFACE:-Wl,-rpath,$<TARGET_FILE_DIR:MQT::CoreQDMINaDevice>>
      $<BUILD_INTERFACE:-Wl,-rpath,$<TARGET_FILE_DIR:MQT::CoreQDMIScDevice>>
      $<BUILD_INTERFACE:-Wl,-rpath,$<TARGET_FILE_DIR:MQT::CoreQDMI_DDSIM_Device>>)
  endif()

  # add to list of MQT core targets
  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
