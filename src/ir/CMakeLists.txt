# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET MQT::CoreIR)
  # collect headers and source files
  file(GLOB_RECURSE IR_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/ir/*.hpp
       ${MQT_CORE_INCLUDE_BUILD_DIR}/ir/*.inc)
  file(GLOB_RECURSE IR_SOURCES **.cpp)

  # create the library target (initially empty)
  add_mqt_core_library(${MQT_CORE_TARGET_NAME}-ir ALIAS_NAME IR)

  # add sources to target
  target_sources(${MQT_CORE_TARGET_NAME}-ir PRIVATE ${IR_SOURCES})

  # add headers using file sets
  target_sources(
    ${MQT_CORE_TARGET_NAME}-ir PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                      FILES ${IR_HEADERS})

  # generate export header
  include(GenerateExportHeader)
  generate_export_header(${MQT_CORE_TARGET_NAME}-ir BASE_NAME mqt_core_ir)
  target_sources(
    ${MQT_CORE_TARGET_NAME}-ir PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/..
                                      FILES ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_ir_export.h)
  if(NOT BUILD_MQT_CORE_SHARED_LIBS)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-ir PUBLIC MQT_CORE_IR_STATIC_DEFINE)
  endif()

  # add to list of MQT core target
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-ir
      PARENT_SCOPE)
endif()
