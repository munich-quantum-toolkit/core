# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")

# add main library code
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# add test code
add_subdirectory(test)
