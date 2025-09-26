# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# add subdirectories
add_subdirectory(device)
add_subdirectory(fomac)
add_subdirectory(zoned)

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-na)
  # collect headers and source files
  file(GLOB NA_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/na/*.hpp
       ${MQT_CORE_INCLUDE_BUILD_DIR}/na/entities/*.hpp
       ${MQT_CORE_INCLUDE_BUILD_DIR}/na/operations/*.hpp)
  file(GLOB NA_SOURCES *.cpp entities/*.cpp operations/*.cpp)

  # create the library target (initially empty)
  add_mqt_core_library(${MQT_CORE_TARGET_NAME}-na ALIAS_NAME NA)

  # add sources to target
  target_sources(${MQT_CORE_TARGET_NAME}-na PRIVATE ${NA_SOURCES})

  # add headers using file sets
  target_sources(
    ${MQT_CORE_TARGET_NAME}-na PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                      FILES ${NA_HEADERS})

  # add link libraries
  target_link_libraries(${MQT_CORE_TARGET_NAME}-na PUBLIC MQT::CoreIR)

  # generate export header
  include(GenerateExportHeader)
  generate_export_header(${MQT_CORE_TARGET_NAME}-na BASE_NAME mqt_core_na)
  target_sources(
    ${MQT_CORE_TARGET_NAME}-na PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/..
                                      FILES ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_na_export.h)
  if(NOT BUILD_MQT_CORE_SHARED_LIBS)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-na PUBLIC MQT_CORE_NA_STATIC_DEFINE)
  endif()

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-na
      PARENT_SCOPE)
endif()
