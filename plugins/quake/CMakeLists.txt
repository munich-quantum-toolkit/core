# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

find_library(CUDAQ_MLIR_RUNTIME_LIB cudaq-mlir-runtime PATHS /workspaces/cuda-quantum/build/lib)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(/workspaces/cuda-quantum/build/include)
include_directories(/workspaces/cuda-quantum/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)
