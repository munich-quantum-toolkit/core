# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

include(CMakeDependentOption)

set(MQT_CORE_SPDLOG_VERSION
    1.17.0
    CACHE STRING "Vendored spdlog version")

cmake_dependent_option(MQT_CORE_SPDLOG_BUILD_SHARED "Build vendored spdlog as shared library" ON
                       "BUILD_MQT_CORE_SHARED_LIBS" OFF)

set(MQT_CORE_SPDLOG_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(MQT_CORE_SPDLOG_INCLUDE_DIR "${MQT_CORE_SPDLOG_ROOT_DIR}/include")
set(MQT_CORE_SPDLOG_SRC_DIR "${MQT_CORE_SPDLOG_ROOT_DIR}/src")
set(MQT_CORE_SPDLOG_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}/mqt-core/third_party/spdlog/include")
set(MQT_CORE_SPDLOG_TARGET "mqt-core-vendor-spdlog")

if(NOT EXISTS "${MQT_CORE_SPDLOG_INCLUDE_DIR}/spdlog/spdlog.h")
  message(FATAL_ERROR "Vendored spdlog header not found at "
                      "${MQT_CORE_SPDLOG_INCLUDE_DIR}/spdlog/spdlog.h")
endif()

set(MQT_CORE_SPDLOG_SOURCES
    ${MQT_CORE_SPDLOG_SRC_DIR}/async.cpp
    ${MQT_CORE_SPDLOG_SRC_DIR}/bundled_fmtlib_format.cpp
    ${MQT_CORE_SPDLOG_SRC_DIR}/cfg.cpp
    ${MQT_CORE_SPDLOG_SRC_DIR}/color_sinks.cpp
    ${MQT_CORE_SPDLOG_SRC_DIR}/file_sinks.cpp
    ${MQT_CORE_SPDLOG_SRC_DIR}/spdlog.cpp
    ${MQT_CORE_SPDLOG_SRC_DIR}/stdout_sinks.cpp)

if(MQT_CORE_SPDLOG_BUILD_SHARED)
  add_library(${MQT_CORE_SPDLOG_TARGET} SHARED ${MQT_CORE_SPDLOG_SOURCES})
else()
  add_library(${MQT_CORE_SPDLOG_TARGET} STATIC ${MQT_CORE_SPDLOG_SOURCES})
endif()

add_library(MQT::spdlog ALIAS ${MQT_CORE_SPDLOG_TARGET})
set_target_properties(${MQT_CORE_SPDLOG_TARGET} PROPERTIES EXPORT_NAME spdlog)

if(NOT TARGET spdlog::spdlog)
  add_library(spdlog::spdlog ALIAS ${MQT_CORE_SPDLOG_TARGET})
endif()

target_include_directories(
  ${MQT_CORE_SPDLOG_TARGET} SYSTEM
  PUBLIC $<BUILD_INTERFACE:${MQT_CORE_SPDLOG_INCLUDE_DIR}>
         $<INSTALL_INTERFACE:${MQT_CORE_SPDLOG_INCLUDE_INSTALL_DIR}>)

target_compile_definitions(${MQT_CORE_SPDLOG_TARGET} PUBLIC SPDLOG_COMPILED_LIB)
if(MQT_CORE_SPDLOG_BUILD_SHARED)
  target_compile_definitions(${MQT_CORE_SPDLOG_TARGET} PUBLIC SPDLOG_SHARED_LIB)
  target_compile_definitions(${MQT_CORE_SPDLOG_TARGET} PRIVATE spdlog_EXPORTS FMT_LIB_EXPORT)
  target_compile_definitions(${MQT_CORE_SPDLOG_TARGET} PUBLIC FMT_SHARED)
endif()

set_target_properties(
  ${MQT_CORE_SPDLOG_TARGET}
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
             ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

find_package(Threads REQUIRED)
target_link_libraries(${MQT_CORE_SPDLOG_TARGET} PUBLIC Threads::Threads)

if(MQT_CORE_INSTALL)
  install(
    TARGETS ${MQT_CORE_SPDLOG_TARGET}
    EXPORT ${MQT_CORE_THIRD_PARTY_TARGETS_EXPORT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
            NAMELINK_COMPONENT ${MQT_CORE_TARGET_NAME}_Development
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Development)

  install(
    DIRECTORY ${MQT_CORE_SPDLOG_INCLUDE_DIR}/
    DESTINATION ${MQT_CORE_SPDLOG_INCLUDE_INSTALL_DIR}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development)

  if(EXISTS "${MQT_CORE_SPDLOG_ROOT_DIR}/LICENSE")
    install(
      FILES ${MQT_CORE_SPDLOG_ROOT_DIR}/LICENSE
      DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/mqt-core/spdlog
      COMPONENT ${MQT_CORE_TARGET_NAME}_Development)
  endif()
endif()
