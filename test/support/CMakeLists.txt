# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_library(MQTCoreTestSupport INTERFACE)
target_include_directories(MQTCoreTestSupport INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(MQTCoreTestSupport INTERFACE MQT::CoreSupport GTest::gtest)
package_add_test(mqt-core-support-test MQT::CoreSupport test_diagnostics.cpp)
