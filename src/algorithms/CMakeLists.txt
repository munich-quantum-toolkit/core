# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-algorithms)
  # collect headers and source files
  file(GLOB_RECURSE ALGO_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/algorithms/**.hpp)
  file(GLOB_RECURSE ALGO_SOURCES **.cpp)

  # create the library target (initially empty)
  add_mqt_core_library(${MQT_CORE_TARGET_NAME}-algorithms)

  # add sources to target
  target_sources(${MQT_CORE_TARGET_NAME}-algorithms PRIVATE ${ALGO_SOURCES})

  # add headers using file sets
  target_sources(
    ${MQT_CORE_TARGET_NAME}-algorithms PUBLIC FILE_SET HEADERS BASE_DIRS
                                              ${MQT_CORE_INCLUDE_BUILD_DIR} FILES ${ALGO_HEADERS})

  # add link libraries
  target_link_libraries(
    ${MQT_CORE_TARGET_NAME}-algorithms
    PUBLIC MQT::CoreIR
    PRIVATE MQT::CoreCircuitOptimizer)

  # generate export header
  include(GenerateExportHeader)
  generate_export_header(${MQT_CORE_TARGET_NAME}-algorithms BASE_NAME mqt_core_algorithms)
  target_sources(
    ${MQT_CORE_TARGET_NAME}-algorithms
    PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/.. FILES
           ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_algorithms_export.h)
  if(NOT BUILD_MQT_CORE_SHARED_LIBS)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-algorithms
                               PUBLIC MQT_CORE_ALGO_STATIC_DEFINE)
  endif()

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-algorithms
      PARENT_SCOPE)
endif()
