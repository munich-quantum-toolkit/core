# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(qtensor_ir_target mqt-core-mlir-unittest-qtensor-ir)
add_executable(${qtensor_ir_target} test_qtensor_ir.cpp)
target_link_libraries(${qtensor_ir_target} PRIVATE GTest::gtest_main MLIRParser MLIRSupportMQT
                                                   MLIRQCOProgramBuilder MLIRQCOPrograms)
mqt_mlir_configure_unittest_target(${qtensor_ir_target})

gtest_discover_tests(${qtensor_ir_target} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
