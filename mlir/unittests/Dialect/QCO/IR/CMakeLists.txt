# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(qco_ir_target mqt-core-mlir-unittest-qco-ir)
add_executable(${qco_ir_target} test_qco_ir.cpp test_qco_ir_matrix.cpp)
target_link_libraries(${qco_ir_target} PRIVATE MLIRParser MLIRSupportMQT GTest::gtest_main
                                               MLIRQCOProgramBuilder MLIRQCOPrograms MQT::CoreDD)
mqt_mlir_configure_unittest_target(${qco_ir_target})

gtest_discover_tests(${qco_ir_target} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
