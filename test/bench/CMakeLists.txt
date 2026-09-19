# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreBench)
  file(GLOB_RECURSE BENCHMARK_TEST_SOURCES *.cpp)
  package_add_test(mqt-core-bench-test MQT::CoreBench ${BENCHMARK_TEST_SOURCES})
  target_include_directories(mqt-core-bench-test PRIVATE ${PROJECT_SOURCE_DIR}/test)
endif()
