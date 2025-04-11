# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# set the include directory for the build tree
set(MQT_MLIR_INCLUDE_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(MQT_MLIR_MIN_VERSION 19.0) # Catalyst supports only 19

# MLIR must be installed on the system
find_package(MLIR REQUIRED CONFIG)
find_package(Catalyst REQUIRED CONFIG)

if(MLIR_VERSION VERSION_LESS MQT_MLIR_MIN_VERSION)
  message(FATAL_ERROR "MLIR version must be at least ${MQT_MLIR_MIN_VERSION}")
endif()
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Add the paths to the MLIR and LLVM CMake modules.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CATALYST_CMAKE_DIR}")

set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIRS};${CATALYST_INCLUDE_DIRS}")

# Include the TableGen, LLVM and MLIR CMake modules.
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MQT_MLIR_INCLUDE_BUILD_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# set the binary directory for the build tree such that, e.g., docs can be generated in the build
# tree
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# add a compile feature for C++17
set(CMAKE_CXX_STANDARD 17)

# add main library code
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# add test code
add_subdirectory(test)
