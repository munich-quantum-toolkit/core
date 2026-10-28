# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_library(MQTCoreTestSupport INTERFACE)
target_sources(
  MQTCoreTestSupport
  INTERFACE
  FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES TestSupport.hpp)
target_include_directories(MQTCoreTestSupport INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(MQTCoreTestSupport INTERFACE LLVMSupport GTest::gtest)
