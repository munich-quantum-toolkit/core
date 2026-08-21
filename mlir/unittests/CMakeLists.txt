# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(MQT_MLIR_UNITTESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(MLIRTestSupport STATIC Support/IRVerification.cpp)
target_include_directories(MLIRTestSupport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  MLIRTestSupport
  PUBLIC MLIRAnalysis
         MLIRIR
         MLIRLLVMDialect
         MLIRQCDialect
         MLIRQCODialect
         MLIRQTensorDialect
         MLIRQTensorUtils
         MLIRSCFDialect
         MLIRSupport
         LLVMSupport)
target_sources(MLIRTestSupport PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} FILES
                                      Support/IRVerification.h)
mqt_mlir_target_use_project_options(MLIRTestSupport)

add_library(MLIRTestCaseUtils INTERFACE)
target_link_libraries(MLIRTestCaseUtils INTERFACE GTest::gtest MLIRIR MLIRSupportMQT
                                                  MLIRTestSupport)
target_sources(MLIRTestCaseUtils INTERFACE FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
                                           FILES TestCaseUtils.h)

add_library(MLIRExactUnitaryTest INTERFACE)
target_link_libraries(MLIRExactUnitaryTest INTERFACE GTest::gtest MLIRFuncDialect MLIRIR
                                                     MLIRQCODDFunctionality MQT::CoreDD)
target_sources(MLIRExactUnitaryTest INTERFACE FILE_SET HEADERS BASE_DIRS
                                              ${CMAKE_CURRENT_SOURCE_DIR} FILES ExactUnitaryTest.h)

function(mqt_mlir_configure_unittest_target target_name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "REQUIRES_EH" "" "")
  target_include_directories(${target_name} PRIVATE ${MQT_MLIR_UNITTESTS_DIR}
                                                    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${target_name} PRIVATE MLIRTestCaseUtils)
  if(arg_REQUIRES_EH)
    set(LLVM_REQUIRES_EH ON)
    set(LLVM_REQUIRES_RTTI ON)
  endif()
  mqt_mlir_target_use_project_options(${target_name})
endfunction()

add_subdirectory(programs)
add_subdirectory(benchmark)
add_subdirectory(Compiler)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Target)
