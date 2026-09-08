# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.28)

file(WRITE "${TEST_BINARY_DIR}/main.cpp" "int main() { return 0; }\n")
file(
  WRITE "${TEST_BINARY_DIR}/CMakeLists.txt"
  [=[
cmake_minimum_required(VERSION 3.28)
project(IPOProbe LANGUAGES CXX)
include("${MQT_SOURCE_DIR}/cmake/StandardProjectSettings.cmake")
add_executable(probe main.cpp)
get_target_property(actual probe INTERPROCEDURAL_OPTIMIZATION)
if(NOT ENABLE_IPO AND actual)
  message(FATAL_ERROR "Disabling ENABLE_IPO left target IPO enabled")
endif()
if(ENABLE_IPO AND ipo_supported AND NOT actual)
  message(FATAL_ERROR "Enabling supported IPO did not enable target IPO")
endif()
]=])

foreach(enabled IN ITEMS ON OFF)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -S "${TEST_BINARY_DIR}" -B "${TEST_BINARY_DIR}/build" -G
      "${TEST_GENERATOR}" "-DCMAKE_CXX_COMPILER=${TEST_CXX_COMPILER}"
      "-DMQT_SOURCE_DIR=${MQT_SOURCE_DIR}" -DCMAKE_BUILD_TYPE=Release "-DENABLE_IPO=${enabled}"
      COMMAND_ERROR_IS_FATAL ANY)
endforeach()
