# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-mlir-compiler-pipeline-test test_compiler_pipeline.cpp)

target_link_libraries(
  mqt-core-mlir-compiler-pipeline-test
  PRIVATE GTest::gtest_main MQTCompilerPipeline MLIRQuartzTranslation MLIRQuartzProgramBuilder
          MLIRParser MQT::CoreIR)

gtest_discover_tests(mqt-core-mlir-compiler-pipeline-test)
