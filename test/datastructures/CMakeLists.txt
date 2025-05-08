# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreDS)
  file(GLOB_RECURSE DS_TEST_SOURCES *.cpp)
  package_add_test(mqt-core-ds-test MQT::CoreDS ${DS_TEST_SOURCES})
endif()
