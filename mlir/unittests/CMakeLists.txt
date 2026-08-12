# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(MQT_MLIR_UNITTESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(MLIRTestCaseUtils INTERFACE)
target_link_libraries(MLIRTestCaseUtils INTERFACE GTest::gtest MLIRIR MLIRSupportMQT)
target_sources(MLIRTestCaseUtils INTERFACE FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
                                           FILES TestCaseUtils.h)

add_library(MLIRExactUnitaryTest INTERFACE)
target_link_libraries(MLIRExactUnitaryTest INTERFACE GTest::gtest MLIRFuncDialect MLIRIR
                                                     MLIRQCODDFunctionality MQT::CoreDD)
target_sources(MLIRExactUnitaryTest INTERFACE FILE_SET HEADERS BASE_DIRS
                                              ${CMAKE_CURRENT_SOURCE_DIR} FILES ExactUnitaryTest.h)

function(mqt_mlir_configure_unittest_target target_name)
  target_include_directories(${target_name} PRIVATE ${MQT_MLIR_UNITTESTS_DIR}
                                                    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${target_name} PRIVATE MLIRTestCaseUtils)
  mqt_mlir_target_use_project_options(${target_name})
endfunction()

add_subdirectory(programs)
add_subdirectory(Compiler)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Target)
