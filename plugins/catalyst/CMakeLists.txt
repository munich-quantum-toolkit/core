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
  mqt-catalyst-plugin
  # todo: for now I fixed the version to 0.1.0 but it should probably be set automatically like in
  # mqt-core. todo: What should be version of this project, equal to mqt-core?
  VERSION 0.1.0
  LANGUAGES C CXX
  DESCRIPTION "MQT's Catalyst Plugin")

# Add path for custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

include(../../cmake/StandardProjectSettings.cmake)
include(../../cmake/PreventInSourceBuilds.cmake)
include(../../cmake/PackageAddTest.cmake)
include(../../cmake/Cache.cmake)

# check if this is the master project or used via add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(MQT_CATALYST_MASTER_PROJECT ON)
else()
  set(MQT_CATALYST_MASTER_PROJECT OFF)
endif()

option(MQT_CATALYST_INSTALL "Generate installation instructions for MQT's Catalyst Plugin"
       ${MQT_CATALYST_MASTER_PROJECT})
option(BUILD_MQT_CATALYST_TESTS "Also build tests for the MQT's Catalyst Plugin project"
       ${MQT_CATALYST_MASTER_PROJECT})

include(../../cmake/ExternalDependencies.cmake)

# set the include directory for the build tree
set(MQT_CATALYST_INCLUDE_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/mqt-catalyst-plugin")

# if the plugin is build as the master project, add the mlir directory, which is already included
# otherwise
if(MQT_CATALYST_MASTER_PROJECT)
  add_subdirectory(../../mlir ${CMAKE_CURRENT_BINARY_DIR}/mlir)
endif()

# add main library code
add_subdirectory(src)

if(MQT_CATALYST_MASTER_PROJECT)
  if(NOT TARGET mqt-catalyst-plugin-uninstall)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/cmake_uninstall.cmake.in
                   ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake IMMEDIATE @ONLY)
    add_custom_target(mqt-catalyst-plugin-uninstall
                      COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
  endif()
else()
  set(mqt-catalyst-plugin_FOUND
      TRUE
      CACHE INTERNAL "True if mqt-catalyst-plugin is found on the system")
endif()
