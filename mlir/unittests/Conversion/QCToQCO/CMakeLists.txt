# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB_RECURSE TEST_SOURCES "*.cpp")

set(target_name mqt-core-mlir-unittest-qc-to-qco)
add_executable(${target_name} ${TEST_SOURCES} test_qc_to_qco.h)

target_link_libraries(
  ${target_name}
  PRIVATE MLIRParser
          MLIRSupportMQT
          GTest::gtest_main
          MLIRQCProgramBuilder
          MLIRQCOProgramBuilder
          MLIRQCPrograms
          MLIRQCOPrograms
          QCToQCO
          MQT::ProjectOptions)

target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

gtest_discover_tests(${target_name})
