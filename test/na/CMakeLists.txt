# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreNA)
  file(GLOB NA_TEST_SOURCES *.cpp)
  package_add_test(mqt-core-na-test MQT::CoreNA ${NA_TEST_SOURCES})
  target_link_libraries(mqt-core-na-test PRIVATE MQT::CoreQASM)
endif()

add_subdirectory(device)
add_subdirectory(fomac)
