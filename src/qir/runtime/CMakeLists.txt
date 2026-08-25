# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qir-runtime)

if(NOT TARGET ${TARGET_NAME})
  # Add the internal QIR runtime library.
  add_library(${TARGET_NAME} STATIC)

  # Add sources to target
  target_sources(
    ${TARGET_NAME}
    PRIVATE QIR.cpp Runtime.cpp ${PROJECT_SOURCE_DIR}/src/qir/include/qir/Definitions.hpp
            ${PROJECT_SOURCE_DIR}/src/qir/include/qir/runtime/QIR.h
            ${PROJECT_SOURCE_DIR}/src/qir/include/qir/runtime/Runtime.hpp)
  target_include_directories(${TARGET_NAME}
                             PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/qir/include>)
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_20)
  set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # Add link libraries
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC MQT::CoreIR MQT::CoreDD
    PRIVATE MQT::ProjectWarnings MQT::ProjectOptions)
endif()
