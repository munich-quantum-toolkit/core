# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-mlir-unittests-benchmark test_benchmark_jeff.cpp
                                                 test_benchmark_pipeline.cpp)

target_link_libraries(mqt-core-mlir-unittests-benchmark PRIVATE GTest::gtest_main
                                                                MQTBenchmarkGenerate)

mqt_mlir_configure_unittest_target(mqt-core-mlir-unittests-benchmark)

gtest_discover_tests(mqt-core-mlir-unittests-benchmark PROPERTIES LABELS mqt-mlir-unittests
                                                                  DISCOVERY_TIMEOUT 60)
