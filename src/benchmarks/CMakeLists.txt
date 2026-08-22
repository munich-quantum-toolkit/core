# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-benchmarks)
  file(GLOB_RECURSE BENCHMARK_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/benchmarks/*.hpp)
  file(GLOB_RECURSE BENCHMARK_SOURCES *.cpp)

  add_mqt_core_library(${MQT_CORE_TARGET_NAME}-benchmarks ALIAS_NAME Benchmarks)

  target_sources(${MQT_CORE_TARGET_NAME}-benchmarks PRIVATE ${BENCHMARK_SOURCES})
  target_link_libraries(${MQT_CORE_TARGET_NAME}-benchmarks
                        PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
  target_sources(
    ${MQT_CORE_TARGET_NAME}-benchmarks
    PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR} FILES ${BENCHMARK_HEADERS})

  include(GenerateExportHeader)
  generate_export_header(${MQT_CORE_TARGET_NAME}-benchmarks BASE_NAME mqt_core_benchmarks)
  target_sources(
    ${MQT_CORE_TARGET_NAME}-benchmarks
    PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/.. FILES
           ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_benchmarks_export.h)
  if(NOT BUILD_MQT_CORE_SHARED_LIBS)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-benchmarks
                               PUBLIC MQT_CORE_BENCHMARKS_STATIC_DEFINE)
  endif()

  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-benchmarks
      PARENT_SCOPE)
endif()
