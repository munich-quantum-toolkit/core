# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-mlir-unittests-benchmark
               test_benchmark_generate.cpp test_benchmark_jeff.cpp test_benchmark_pipeline.cpp)

target_link_libraries(mqt-core-mlir-unittests-benchmark PRIVATE GTest::gtest_main
                                                                MQTBenchmarkGenerate)

mqt_mlir_configure_unittest_target(mqt-core-mlir-unittests-benchmark)

gtest_discover_tests(mqt-core-mlir-unittests-benchmark PROPERTIES LABELS mqt-mlir-unittests
                                                                  DISCOVERY_TIMEOUT 60)

add_test(
  NAME mqt-core-mlir-benchmark-cli
  COMMAND
    ${CMAKE_COMMAND} "-DCLI=$<TARGET_FILE:mqt-core-bench>"
    "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/cli-output-$<CONFIG>" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/test_benchmark_cli.cmake")
set_tests_properties(mqt-core-mlir-benchmark-cli PROPERTIES LABELS mqt-mlir-unittests)
