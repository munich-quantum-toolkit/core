# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

include(cmake/ExternalDependencies.cmake)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CATALYST_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(include)
add_subdirectory(src)
