# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-dd)
  # collect headers and source files
  file(GLOB_RECURSE DD_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/dd/*.hpp)
  file(GLOB_RECURSE DD_SOURCES **.cpp)

  # create the library target (initially empty)
  add_library(${MQT_CORE_TARGET_NAME}-dd)

  # add sources to target
  target_sources(${MQT_CORE_TARGET_NAME}-dd PRIVATE ${DD_SOURCES})

  # add headers using file sets
  target_sources(
    ${MQT_CORE_TARGET_NAME}-dd PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                      FILES ${DD_HEADERS})

  # add link libraries
  target_link_libraries(
    ${MQT_CORE_TARGET_NAME}-dd
    PUBLIC MQT::CoreIR nlohmann_json
    PRIVATE MQT::ProjectOptions MQT::ProjectWarnings)

  # add MQT alias
  add_library(MQT::CoreDD ALIAS ${MQT_CORE_TARGET_NAME}-dd)

  # set versioning information
  set_target_properties(
    mqt-core-dd
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreDD)

  # generate export header
  include(GenerateExportHeader)
  generate_export_header(${MQT_CORE_TARGET_NAME}-dd BASE_NAME mqt_core_dd)
  target_sources(
    ${MQT_CORE_TARGET_NAME}-dd PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/..
                                      FILES ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_dd_export.h)
  if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-dd PUBLIC MQT_CORE_DD_STATIC_DEFINE)
  endif()

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-dd
      PARENT_SCOPE)
endif()
