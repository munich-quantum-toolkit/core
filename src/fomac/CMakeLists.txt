# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-fomac)

if(NOT TARGET ${TARGET_NAME})
  # Add FoMaC library
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME FoMaC)

  # Add sources to target
  target_sources(${TARGET_NAME} PRIVATE FoMaC.cpp)

  # Add headers using file sets
  target_sources(${TARGET_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                       FILES ${MQT_CORE_INCLUDE_BUILD_DIR}/fomac/FoMaC.hpp)

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
