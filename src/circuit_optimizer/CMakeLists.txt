# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-circuit-optimizer)
  # collect headers and source files
  file(GLOB_RECURSE CIRCUIT_OPTIMIZER_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/circuit_optimizer/*.hpp)
  file(GLOB_RECURSE CIRCUIT_OPTIMIZER_SOURCES **.cpp)

  # create the library target (initially empty)
  add_mqt_core_library(${MQT_CORE_TARGET_NAME}-circuit-optimizer)

  # add sources to target
  target_sources(${MQT_CORE_TARGET_NAME}-circuit-optimizer PRIVATE ${CIRCUIT_OPTIMIZER_SOURCES})

  # add headers using file sets
  target_sources(
    ${MQT_CORE_TARGET_NAME}-circuit-optimizer
    PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR} FILES
           ${CIRCUIT_OPTIMIZER_HEADERS})

  # add link libraries
  target_link_libraries(
    ${MQT_CORE_TARGET_NAME}-circuit-optimizer
    PUBLIC MQT::CoreIR
    PRIVATE MQT::ProjectOptions MQT::ProjectWarnings)

  # set versioning information
  set_target_properties(
    ${MQT_CORE_TARGET_NAME}-circuit-optimizer
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreCircuitOptimizer)

  # generate export header
  include(GenerateExportHeader)
  generate_export_header(${MQT_CORE_TARGET_NAME}-circuit-optimizer BASE_NAME
                         mqt_core_circuit_optimizer)
  target_sources(
    ${MQT_CORE_TARGET_NAME}-circuit-optimizer
    PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/.. FILES
           ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_circuit_optimizer_export.h)
  if(NOT BUILD_MQT_CORE_SHARED_LIBS)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-circuit-optimizer
                               PUBLIC MQT_CORE_CIRCUIT_OPTIMIZER_STATIC_DEFINE)
  endif()

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-circuit-optimizer
      PARENT_SCOPE)
endif()
