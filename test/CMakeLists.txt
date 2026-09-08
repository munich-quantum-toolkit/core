# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# add various subdirectories containing tests
add_subdirectory(bench)
add_subdirectory(dd)
add_subdirectory(qdmi)

# copy test circuits to build directory
file(COPY ${PROJECT_SOURCE_DIR}/test/circuits DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_test(
  NAME cmake_ipo_toggle
  COMMAND
    "${CMAKE_COMMAND}" "-DMQT_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/ipo-probe" "-DTEST_GENERATOR=${CMAKE_GENERATOR}"
    "-DTEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_ipo.cmake")
