# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qir-runtime)

if(NOT TARGET ${TARGET_NAME})
  # Add internal QIR runtime library
  add_library(${TARGET_NAME} STATIC QIR.cpp Runtime.cpp)

  target_include_directories(${TARGET_NAME}
                             PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/qir/include>)
  set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # Add link libraries
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC MQT::CoreIR MQT::CoreDD
    PRIVATE MQT::ProjectWarnings MQT::ProjectOptions)
endif()
