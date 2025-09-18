# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreCircuitOptimizer)
  file(GLOB_RECURSE CIRCUIT_OPTIMIZER_TEST_SOURCES *.cpp)
  package_add_test(mqt-core-circuit-optimizer-test MQT::CoreCircuitOptimizer
                   ${CIRCUIT_OPTIMIZER_TEST_SOURCES})
  target_link_libraries(mqt-core-circuit-optimizer-test PRIVATE MQT::CoreAlgorithms MQT::CoreQASM)
endif()
