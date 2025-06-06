# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-driver)

if(NOT TARGET ${TARGET_NAME})

  add_mqt_core_library(${TARGET_NAME})
  target_sources(${TARGET_NAME} PRIVATE driver.cpp)
  target_sources(${TARGET_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                       FILES ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/driver.hpp)
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi MQT::CoreNaQDMI
    PRIVATE qdmi::project_warnings MQT::ProjectOptions MQT::ProjectWarnings)
  set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

endif()
