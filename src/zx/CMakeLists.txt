# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-zx)
  # make sure the MQT::multiprecision target is available
  if(NOT TARGET MQT::Multiprecision)
    # handle the case for a complete Boost installation
    if(NOT TARGET Boost::multiprecision)
      add_library(multiprecision INTERFACE)
      target_link_libraries(multiprecision INTERFACE Boost::headers)
      add_library(MQT::Multiprecision ALIAS multiprecision)
      set(MQT_CORE_ZX_SYSTEM_BOOST
          TRUE
          CACHE BOOL "Whether a system version of Boost was used")
    else()
      # Boost::multiprecision does not provide its own installation instructions. As a consequence,
      # this needs a workaround.

      # first, we get the include directory of the multiprecision library
      get_target_property(MULTIPRECISION_INCLUDE_DIR Boost::multiprecision
                          INTERFACE_INCLUDE_DIRECTORIES)

      # then, we create a target that will be used to export the include directory
      add_library(multiprecision INTERFACE)
      target_include_directories(multiprecision SYSTEM
                                 INTERFACE $<BUILD_INTERFACE:${MULTIPRECISION_INCLUDE_DIR}>)
      add_library(MQT::Multiprecision ALIAS multiprecision)

      # add headers using file sets
      file(GLOB_RECURSE MULTIPRECISION_HEADERS ${MULTIPRECISION_INCLUDE_DIR}/*.hpp)

      target_sources(multiprecision PUBLIC FILE_SET HEADERS BASE_DIRS ${MULTIPRECISION_INCLUDE_DIR}
                                           FILES ${MULTIPRECISION_HEADERS})

      set(MQT_CORE_ZX_SYSTEM_BOOST
          FALSE
          CACHE BOOL "Whether a system version of Boost was used")
    endif()
  endif()

  # collect headers and source files
  file(GLOB_RECURSE ZX_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/zx/*.hpp)
  file(GLOB_RECURSE ZX_SOURCES **.cpp)

  # create the library target (initially empty)
  add_mqt_core_library(${MQT_CORE_TARGET_NAME}-zx ALIAS_NAME ZX)

  # add sources to target
  target_sources(${MQT_CORE_TARGET_NAME}-zx PRIVATE ${ZX_SOURCES})

  # add headers using file sets
  target_sources(
    ${MQT_CORE_TARGET_NAME}-zx PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                      FILES ${ZX_HEADERS})

  # add link libraries
  target_link_libraries(
    ${MQT_CORE_TARGET_NAME}-zx
    PUBLIC MQT::CoreIR MQT::Multiprecision
    PRIVATE MQT::ProjectOptions MQT::ProjectWarnings)

  option(MQT_CORE_WITH_GMP "Whether to use GMP for multiprecision arithmetic" OFF)
  if(MQT_CORE_WITH_GMP)
    find_package(GMP REQUIRED)
    # link to GMP libraries if present
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-zx PUBLIC GMP)
    target_link_libraries(${MQT_CORE_TARGET_NAME}-zx PUBLIC GMP::gmp GMP::gmpxx)
  endif()

  # set versioning information
  set_target_properties(
    ${MQT_CORE_TARGET_NAME}-zx
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreZX)

  # generate export header
  include(GenerateExportHeader)
  generate_export_header(${MQT_CORE_TARGET_NAME}-zx BASE_NAME mqt_core_zx)
  target_sources(
    ${MQT_CORE_TARGET_NAME}-zx PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/..
                                      FILES ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_zx_export.h)
  if(NOT BUILD_MQT_CORE_SHARED_LIBS)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-zx PUBLIC MQT_CORE_ZX_STATIC_DEFINE)
  endif()

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-zx multiprecision
      PARENT_SCOPE)
endif()
