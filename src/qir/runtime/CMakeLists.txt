# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qir-runtime)

if(NOT TARGET ${TARGET_NAME})
  # Add QIRBackend library
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME QIRRuntime)

  # Add sources to target
  target_sources(${TARGET_NAME} PRIVATE QIR.cpp Runtime.cpp
                                        ${MQT_CORE_INCLUDE_BUILD_DIR}/qir/runtime/Runtime.hpp)

  # Add headers using file sets
  target_sources(${TARGET_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                       FILES ${MQT_CORE_INCLUDE_BUILD_DIR}/qir/runtime/QIR.h)

  # Add link libraries
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC MQT::CoreIR MQT::CoreDD
    PRIVATE MQT::ProjectWarnings MQT::ProjectOptions)

  # Set versioning information
  set_target_properties(
    ${TARGET_NAME}
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreQIRRuntime)

  # Set c++ standard
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
endif()
