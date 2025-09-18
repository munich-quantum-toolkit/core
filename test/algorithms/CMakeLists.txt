# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreAlgorithms)
  file(GLOB_RECURSE ALGO_TEST_SOURCES *.cpp)
  package_add_test(mqt-core-algorithms-test MQT::CoreAlgorithms ${ALGO_TEST_SOURCES})
  target_link_libraries(mqt-core-algorithms-test PRIVATE MQT::CoreDD MQT::CoreCircuitOptimizer)
endif()
