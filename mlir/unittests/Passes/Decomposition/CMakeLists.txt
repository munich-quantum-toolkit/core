# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(testname "mqt-core-mlir-decomposition-test")
file(GLOB_RECURSE DECOMPOSITION_TEST_SOURCES *.cpp)

if(NOT TARGET ${testname})
  # create an executable in which the tests will be stored
  add_executable(${testname} ${DECOMPOSITION_TEST_SOURCES})
  # link the Google test infrastructure and a default main function to the test executable.
  target_link_libraries(
    ${testname}
    PRIVATE GTest::gmock
            GTest::gtest_main
            LLVMFileCheck
            MLIRPass
            MLIRTransforms
            MQTCompilerPipeline
            MQT::CoreIR
            Eigen3::Eigen
            QcoPasses)
  # fix alignment issues for stack-allocated fixed-size matrices
  target_compile_definitions(
    ${testname}
    PRIVATE EIGEN_MAX_STATIC_ALIGN_BYTES=16)
  # discover tests
  gtest_discover_tests(${testname} DISCOVERY_TIMEOUT 60)
  set_target_properties(${testname} PROPERTIES FOLDER unittests)
endif()
