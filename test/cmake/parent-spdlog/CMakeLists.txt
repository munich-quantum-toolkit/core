# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.24...4.2)

project(mqt-core-parent-spdlog LANGUAGES C CXX)

if(NOT MQT_CORE_SOURCE_DIR)
  message(FATAL_ERROR "MQT_CORE_SOURCE_DIR is required")
endif()

add_library(spdlog::spdlog INTERFACE IMPORTED)

foreach(option IN ITEMS MQT_CORE_INSTALL BUILD_MQT_CORE_BINDINGS BUILD_MQT_CORE_TESTS
                        BUILD_MQT_CORE_DOCUMENTATION)
  set(${option}
      OFF
      CACHE BOOL "")
endforeach()
add_subdirectory("${MQT_CORE_SOURCE_DIR}" mqt-core EXCLUDE_FROM_ALL)
