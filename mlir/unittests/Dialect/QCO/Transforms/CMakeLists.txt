# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-mlir-dialect-qco-transforms-test test_qco_quaternion_merge.cpp)

target_link_libraries(
  mqt-core-mlir-dialect-qco-transforms-test
  PRIVATE GTest::gtest_main
          MLIRQCOProgramBuilder
          MLIRQCOTransforms
          MLIRIR
          MLIRPass
          MLIRSupport
          LLVMSupport)

gtest_discover_tests(mqt-core-mlir-dialect-qco-transforms-test)
