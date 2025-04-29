# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# add various subdirectories containing tests
add_subdirectory(algorithms)
add_subdirectory(circuit_optimizer)
add_subdirectory(datastructures)
add_subdirectory(dd)
add_subdirectory(ir)
add_subdirectory(na)
add_subdirectory(zx)

# copy test circuits to build directory
file(COPY ${PROJECT_SOURCE_DIR}/test/circuits DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
