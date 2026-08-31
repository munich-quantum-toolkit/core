# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# set required cmake version
cmake_minimum_required(VERSION 3.24...4.4)

project(
  mqt-core
  LANGUAGES C CXX
  DESCRIPTION "MQT Core - The Backbone of the Munich Quantum Toolkit")

if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 20)
  set(CMAKE_CXX_STANDARD
      20
      CACHE STRING "C++ standard to conform to" FORCE)
endif()

# Add path for custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(AddMQTPythonBinding)
include(StandardProjectSettings)
include(PreventInSourceBuilds)
include(PackageAddTest)
include(Cache)
include(AddMQTCoreLibrary)
include(AddMQTQDMIDevice)

option(BUILD_MQT_CORE_BINDINGS "Build the MQT Core Python bindings" OFF)
if(BUILD_MQT_CORE_BINDINGS)
  # ensure that the BINDINGS option is set
  set(BINDINGS
      ON
      CACHE INTERNAL "Enable settings related to Python bindings")
  # Some common settings for finding Python
  set(Python_FIND_VIRTUALENV
      FIRST
      CACHE STRING "Give precedence to virtualenvs when searching for Python")
  set(Python_FIND_FRAMEWORK
      LAST
      CACHE STRING "Prefer Brew/Conda to Apple framework Python")
  set(Python_ARTIFACTS_INTERACTIVE
      ON
      CACHE BOOL "Prevent multiple searches for Python and instead cache the results.")

  # top-level call to find Python
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module
                                               ${SKBUILD_SABI_COMPONENT})
endif()

option(MQT_CORE_INSTALL "Generate installation instructions for MQT Core" ${PROJECT_IS_TOP_LEVEL})
option(BUILD_MQT_CORE_TESTS "Build tests for the MQT Core project" ${PROJECT_IS_TOP_LEVEL})
option(BUILD_MQT_CORE_SHARED_LIBS "Build MQT Core libraries as shared libraries"
       ${BUILD_SHARED_LIBS})
option(BUILD_MQT_CORE_QDMI_DDSIM_DEVICE "Build the MQT Core DDSIM QDMI device"
       ${PROJECT_IS_TOP_LEVEL})
option(BUILD_MQT_CORE_QDMI_SC_DEVICE "Build the MQT Core superconducting QDMI device"
       ${PROJECT_IS_TOP_LEVEL})
option(BUILD_MQT_CORE_DOCUMENTATION "Generate documentation artifacts as part of the CMake build"
       OFF)
if(BUILD_MQT_CORE_TESTS AND (NOT BUILD_MQT_CORE_QDMI_DDSIM_DEVICE OR NOT
                                                                     BUILD_MQT_CORE_QDMI_SC_DEVICE))
  message(FATAL_ERROR "BUILD_MQT_CORE_TESTS requires all bundled QDMI devices")
endif()
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(
    FATAL_ERROR
      "MQT Core cannot be built with GCC on macOS because its required MLIR dependency uses an incompatible C++ standard library"
  )
endif()
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "17.0.0")
    message(
      FATAL_ERROR
        "MQT Core requires AppleClang 17 or newer because its required MLIR dependency needs complete C++20 support"
    )
  endif()
endif()

# try to determine the project version
include(GetVersion)
get_mqt_core_version()

project(
  mqt-core
  LANGUAGES C CXX
  VERSION ${MQT_CORE_VERSION}
  DESCRIPTION "MQT Core - The Backbone of the Munich Quantum Toolkit")

set(MQT_CORE_TARGET_NAME "mqt-core")

include(SetupMLIR)

include(cmake/ExternalDependencies.cmake)

# set the include directory for the build tree
set(MQT_CORE_INCLUDE_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/mqt-core")

if(MQT_CORE_INSTALL)
  if(APPLE)
    set(BASEPOINT @loader_path)
  else()
    set(BASEPOINT $ORIGIN)
  endif()
  set(CMAKE_INSTALL_RPATH ${BASEPOINT} ${BASEPOINT}/${CMAKE_INSTALL_LIBDIR}
                          ${BASEPOINT}/../${CMAKE_INSTALL_LIBDIR})
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# add main library code
add_subdirectory(src)

# add bindings code if enabled
if(BUILD_MQT_CORE_BINDINGS)
  add_subdirectory(bindings)
endif()

# add test code
if(BUILD_MQT_CORE_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()

add_subdirectory(mlir)

# Preserve the top-level `mlir-doc` behavior without modifying a target owned by an embedding
# project.
if(PROJECT_IS_TOP_LEVEL)
  add_custom_command(
    TARGET mlir-doc
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/docs/
            ${CMAKE_CURRENT_SOURCE_DIR}/docs/mlir/
    COMMAND ${CMAKE_COMMAND} -D DOCS_DIR:PATH=${CMAKE_CURRENT_SOURCE_DIR}/docs/mlir -P
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CleanMLIRDocs.cmake
    COMMENT "Copying and cleaning up generated MLIR documentation"
    VERBATIM)
endif()

if(BUILD_MQT_CORE_DOCUMENTATION)
  if(PROJECT_IS_TOP_LEVEL)
    add_custom_target(mqt-core-docs DEPENDS mlir-doc)
  else()
    add_custom_target(
      mqt-core-docs
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/docs/
              ${CMAKE_CURRENT_SOURCE_DIR}/docs/mlir/
      COMMAND ${CMAKE_COMMAND} -D DOCS_DIR:PATH=${CMAKE_CURRENT_SOURCE_DIR}/docs/mlir -P
              ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CleanMLIRDocs.cmake
      DEPENDS mlir-doc
      COMMENT "Copying and cleaning up generated MLIR documentation"
      VERBATIM)
  endif()
  foreach(binding ir dd qdmi)
    add_dependencies(${MQT_CORE_TARGET_NAME}-${binding}-bindings mqt-core-docs)
  endforeach()
endif()

if(BUILD_MQT_CORE_BINDINGS)
  set(MQT_CORE_WHEEL_TARGETS mqt-core-ir mqt-core-dd mqt-core-ir-bindings mqt-core-dd-bindings
                             mqt-core-qdmi-bindings mqt-core-mlir-bindings)
  if(BUILD_MQT_CORE_QDMI_DDSIM_DEVICE)
    list(APPEND MQT_CORE_WHEEL_TARGETS mqt-core-qdmi-ddsim-device)
  endif()
  if(BUILD_MQT_CORE_QDMI_SC_DEVICE)
    list(APPEND MQT_CORE_WHEEL_TARGETS mqt-core-qdmi-sc-device)
  endif()
  add_custom_target(mqt-core-wheel)
  add_dependencies(mqt-core-wheel ${MQT_CORE_WHEEL_TARGETS})
endif()

if(PROJECT_IS_TOP_LEVEL)
  if(NOT TARGET mqt-core-uninstall)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
                   ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake IMMEDIATE @ONLY)
    add_custom_target(mqt-core-uninstall COMMAND ${CMAKE_COMMAND} -P
                                                 ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
  endif()
else()
  set(mqt-core_FOUND
      TRUE
      CACHE INTERNAL "True if mqt-core is found on the system")
endif()
