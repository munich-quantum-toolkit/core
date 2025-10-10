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
  mqt-core-catalyst-plugin
  LANGUAGES C CXX
  VERSION 0.1.0
  DESCRIPTION "MQT Core MLIR Plugin for Catalyst")

set(CMAKE_CXX_STANDARD
    20
    CACHE STRING "C++ standard to conform to")

if(ENABLE_COVERAGE)
  add_compile_options(--coverage -O0)
  add_link_options(--coverage)
endif()

include(cmake/ExternalDependencies.cmake)

set(MQT_MLIR_PLUGIN_SOURCE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(MQT_MLIR_PLUGIN_BUILD_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")

# MLIR must be installed on the system
find_package(MLIR REQUIRED CONFIG)

# Add the paths to the MLIR and LLVM CMake modules.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include the TableGen, LLVM and MLIR CMake modules.
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(BEFORE ${MQT_MLIR_PLUGIN_SOURCE_INCLUDE_DIR})
include_directories(BEFORE ${MQT_MLIR_PLUGIN_BUILD_INCLUDE_DIR})
include_directories(${CATALYST_INCLUDE_DIRS})

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

if(APPLE)
  set(BASEPOINT @loader_path)
else()
  set(BASEPOINT $ORIGIN)
endif()
set(CMAKE_INSTALL_RPATH ${BASEPOINT} ${BASEPOINT}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
