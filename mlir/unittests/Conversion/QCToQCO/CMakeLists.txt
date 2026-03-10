# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(target_name mqt-core-mlir-unittest-qc-to-qco)
add_executable(${target_name} test_qc_to_qco.cpp)

target_link_libraries(
  ${target_name}
  PRIVATE MLIRParser
          MLIRSupportMQT
          GTest::gtest_main
          MLIRQCProgramBuilder
          MLIRQCOProgramBuilder
          MLIRQCPrograms
          MLIRQCOPrograms
          QCToQCO)

mqt_mlir_configure_unittest_target(${target_name})

gtest_discover_tests(${target_name} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
