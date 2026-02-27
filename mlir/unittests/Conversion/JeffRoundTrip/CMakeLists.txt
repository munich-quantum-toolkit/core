# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(target_name mqt-core-mlir-unittest-jeff-round-trip)
add_executable(${target_name} test_jeff_round_trip.cpp)

target_link_libraries(
  ${target_name}
  PRIVATE GTest::gtest_main
          MLIRArithDialect
          MLIRParser
          MLIRPass
          MLIRQCOPrograms
          MLIRQCOProgramBuilder
          MLIRSupportMQT
          MLIRTransforms
          JeffToQCO
          QCOToJeff)

target_include_directories(${target_name} PRIVATE ${jeff-mlir_SOURCE_DIR}/include
                                                  ${jeff-mlir_BINARY_DIR}/include)

mqt_mlir_configure_unittest_target(${target_name})

gtest_discover_tests(${target_name} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
