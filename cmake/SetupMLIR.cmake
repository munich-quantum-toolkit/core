# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# set the include directory for the build tree
set(MQT_MLIR_SOURCE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/mlir/include")
set(MQT_MLIR_BUILD_INCLUDE_DIR "${PROJECT_BINARY_DIR}/mlir/include")
set(MQT_MLIR_MIN_VERSION 21.0)

# MLIR must be installed on the system
find_package(MLIR REQUIRED CONFIG)
if(MLIR_VERSION VERSION_LESS MQT_MLIR_MIN_VERSION)
  message(FATAL_ERROR "MLIR version must be at least ${MQT_MLIR_MIN_VERSION}")
endif()
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

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
include_directories(${MQT_MLIR_SOURCE_INCLUDE_DIR})
include_directories(${MQT_MLIR_BUILD_INCLUDE_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

string(REPLACE "." ";" MLIR_VERSION_COMPONENTS ${MLIR_VERSION})
list(GET MLIR_VERSION_COMPONENTS 0 MLIR_VERSION_MAJOR)
add_compile_definitions(MLIR_VERSION_MAJOR=${MLIR_VERSION_MAJOR})

# set the binary directory for the build tree such that, e.g., docs can be generated in the build
# tree
set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
