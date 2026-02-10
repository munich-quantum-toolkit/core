# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-mlir-qco-dialect-test test_unitary_matrix.cpp test_unitary_op_interface.cpp)

target_link_libraries(mqt-core-mlir-qco-dialect-test PRIVATE GTest::gtest_main MLIRQCODialect
                                                             MLIRLLVMDialect MLIRQCOProgramBuilder)

gtest_discover_tests(mqt-core-mlir-qco-dialect-test)

file(
  GLOB_RECURSE
  QCO_IR_TEST_SOURCES
  "QubitManagement/*.cpp"
  "Operations/*.cpp"
  "Operations/StandardGates/*.cpp"
  "Modifiers/test_ctrl_op.cpp"
  "Modifiers/test_inv_op.cpp"
  "test_qco_ir.cpp")

set(qco_ir_target mqt-core-mlir-unittest-qco-ir)
add_executable(${qco_ir_target} ${QCO_IR_TEST_SOURCES} test_qco_ir.h)
target_link_libraries(${qco_ir_target} PRIVATE MLIRParser MLIRSupportMQT GTest::gtest_main
                                               MLIRQCOProgramBuilder)
target_include_directories(${qco_ir_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../programs)
gtest_discover_tests(${qco_ir_target})

add_subdirectory(Modifiers)
