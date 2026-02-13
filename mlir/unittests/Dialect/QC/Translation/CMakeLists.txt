# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB_RECURSE TEST_SOURCES "*.cpp")

set(target_name mqt-core-mlir-unittest-quantum-computation-translation)
add_executable(${target_name} ${TEST_SOURCES})

target_link_libraries(
  ${target_name}
  PRIVATE GTest::gtest_main
          MLIRParser
          MLIRSupportMQT
          MLIRQCProgramBuilder
          MLIRQCTranslation
          MLIRQCPrograms
          MLIRQuantumComputationPrograms
          MQT::CoreIR
          MQT::ProjectOptions)

gtest_discover_tests(${target_name})
