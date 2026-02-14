# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB_RECURSE QCO_IR_TEST_SOURCES "*.cpp")

set(qco_ir_target mqt-core-mlir-unittest-qco-ir)
add_executable(${qco_ir_target} ${QCO_IR_TEST_SOURCES} test_qco_ir.h)
target_link_libraries(${qco_ir_target} PRIVATE MLIRParser MLIRSupportMQT GTest::gtest_main
                                               MLIRQCOProgramBuilder MLIRQCOPrograms MQT::CoreDD)
target_include_directories(${qco_ir_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

mqt_mlir_target_use_project_options(${qco_ir_target})

gtest_discover_tests(${qco_ir_target})
