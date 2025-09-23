# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
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
                                       FILES ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/Driver.hpp)

  # Add link libraries
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi
    PRIVATE MQT::CoreQDMINaDevice qdmi::project_warnings)

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS ${MQT_CORE_TARGETS} ${TARGET_NAME})
endif()

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-fomac)

if(NOT TARGET ${TARGET_NAME})
  # Add FoMaC library
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME QDMIFoMaC)

  # Add sources to target
  target_sources(${TARGET_NAME} PRIVATE FoMaC.cpp)

  # Add headers using file sets
  target_sources(${TARGET_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                       FILES ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/FoMaC.hpp)

  # Add link libraries
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi MQT::CoreQDMIDriver
    PRIVATE spdlog::spdlog)

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${TARGET_NAME}
      PARENT_SCOPE)
endif()
