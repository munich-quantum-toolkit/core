# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-dd)
  file(GLOB_RECURSE DD_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/dd/*.hpp)
  file(GLOB_RECURSE DD_SOURCES **.cpp)

  add_mqt_core_library(${MQT_CORE_TARGET_NAME}-dd ALIAS_NAME DD)

  target_sources(${MQT_CORE_TARGET_NAME}-dd PRIVATE ${DD_SOURCES})

  target_sources(
    ${MQT_CORE_TARGET_NAME}-dd PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                      FILES ${DD_HEADERS})

  target_link_libraries(${MQT_CORE_TARGET_NAME}-dd
                        PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)

  include(GenerateExportHeader)
  generate_export_header(${MQT_CORE_TARGET_NAME}-dd BASE_NAME mqt_core_dd)
  target_sources(
    ${MQT_CORE_TARGET_NAME}-dd PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/..
                                      FILES ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_dd_export.h)
  if(NOT BUILD_MQT_CORE_SHARED_LIBS)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-dd PUBLIC MQT_CORE_DD_STATIC_DEFINE)
  endif()

  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-dd
      PARENT_SCOPE)
endif()
