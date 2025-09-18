# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreZX)
  file(GLOB_RECURSE ZX_TEST_SOURCES *.cpp)
  package_add_test(mqt-core-zx-test MQT::CoreZX ${ZX_TEST_SOURCES})
  target_link_libraries(mqt-core-zx-test PRIVATE MQT::CoreQASM)
endif()
