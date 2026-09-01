# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-bench-library)
  file(GLOB_RECURSE BENCH_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/bench/*.hpp
       ${MQT_CORE_INCLUDE_BUILD_DIR}/bench/*.inc)
  file(GLOB_RECURSE BENCH_SOURCES *.cpp)

  add_mqt_core_library(${MQT_CORE_TARGET_NAME}-bench-library ALIAS_NAME Bench)
  set_target_properties(${MQT_CORE_TARGET_NAME}-bench-library
                        PROPERTIES OUTPUT_NAME ${MQT_CORE_TARGET_NAME}-bench)

  target_sources(${MQT_CORE_TARGET_NAME}-bench-library PRIVATE ${BENCH_SOURCES})
  target_link_libraries(${MQT_CORE_TARGET_NAME}-bench-library
                        PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
  target_sources(
    ${MQT_CORE_TARGET_NAME}-bench-library
    PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR} FILES ${BENCH_HEADERS})

  include(GenerateExportHeader)
  generate_export_header(${MQT_CORE_TARGET_NAME}-bench-library BASE_NAME mqt_core_bench)
  target_sources(
    ${MQT_CORE_TARGET_NAME}-bench-library
    PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/.. FILES
           ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_bench_export.h)
  if(NOT BUILD_MQT_CORE_SHARED_LIBS)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-bench-library
                               PUBLIC MQT_CORE_BENCH_STATIC_DEFINE)
  endif()

  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-bench-library
      PARENT_SCOPE)
endif()
