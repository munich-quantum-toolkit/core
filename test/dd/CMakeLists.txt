# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreDD)
  file(GLOB_RECURSE DD_TEST_SOURCES *.cpp)
  package_add_test(mqt-core-dd-test MQT::CoreDD ${DD_TEST_SOURCES})
  target_link_libraries(mqt-core-dd-test PRIVATE MQT::CoreCircuitOptimizer)
  target_link_libraries(mqt-core-dd-test PRIVATE MQT::CoreQASM)
endif()
