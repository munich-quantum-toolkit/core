# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# set required cmake version
cmake_minimum_required(VERSION 3.24...4.0)

project(
  mqt-core
  LANGUAGES C CXX
  DESCRIPTION "MQT Core - The Backbone of the Munich Quantum Toolkit")

# Add path for custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(StandardProjectSettings)
include(PreventInSourceBuilds)
include(PackageAddTest)
include(Cache)
include(AddMQTCoreLibrary)

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

  if(DISABLE_GIL)
    message(STATUS "Disabling Python GIL")
    add_compile_definitions(Py_GIL_DISABLED)
  endif()

  # top-level call to find Python
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
endif()

# check if this is the master project or used via add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(MQT_CORE_MASTER_PROJECT ON)
else()
  set(MQT_CORE_MASTER_PROJECT OFF)
endif()

option(MQT_CORE_INSTALL "Generate installation instructions for MQT Core"
       ${MQT_CORE_MASTER_PROJECT})
option(BUILD_MQT_CORE_TESTS "Also build tests for the MQT Core project" ${MQT_CORE_MASTER_PROJECT})
option(BUILD_MQT_CORE_SHARED_LIBS "Build MQT Core libraries as shared libraries"
       ${BUILD_SHARED_LIBS})

# try to determine the project version
include(GetVersion)
get_mqt_core_version()

project(
  mqt-core
  LANGUAGES C CXX
  VERSION ${MQT_CORE_VERSION}
  DESCRIPTION "MQT Core - The Backbone of the Munich Quantum Toolkit")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ExternalDependencies)
include(AddMQTCoreBinding)

# set the include directory for the build tree
set(MQT_CORE_INCLUDE_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/mqt-core")

set(MQT_CORE_TARGET_NAME "mqt-core")

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

option(BUILD_MQT_CORE_BENCHMARKS "Also build benchmarks for the MQT Core project" OFF)
if(BUILD_MQT_CORE_BENCHMARKS)
  add_subdirectory(eval)
endif()

option(BUILD_MQT_CORE_MLIR "Build the MLIR submodule of the MQT Core project" OFF)
if(BUILD_MQT_CORE_MLIR)
  include(SetupMLIR)
  add_subdirectory(mlir)
endif()

if(MQT_CORE_MASTER_PROJECT)
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
