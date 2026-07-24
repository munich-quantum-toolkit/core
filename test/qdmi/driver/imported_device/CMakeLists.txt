# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.24)
project(mqt-core-qdmi-imported-device-test LANGUAGES CXX)

if(NOT MQT_CORE_QDMI_DEVICE_TARGETS OR NOT MQT_CORE_QDMI_HELPER)
  message(FATAL_ERROR "Device targets and the QDMI helper are required")
endif()

include("${MQT_CORE_QDMI_DEVICE_TARGETS}")
include("${MQT_CORE_QDMI_HELPER}")

set(device mqt-core-qdmi-metadata-device)
get_target_property(device_id ${device} MQT_QDMI_DEVICE_ID)
get_target_property(device_prefix ${device} MQT_QDMI_DEVICE_PREFIX)
if(NOT device_id STREQUAL "test.metadata-only" OR NOT device_prefix STREQUAL "TEST_METADATA")
  message(FATAL_ERROR "Exported QDMI device metadata was not preserved")
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp" "int main() { return 0; }\n")
add_executable(imported-device-consumer "${CMAKE_CURRENT_BINARY_DIR}/main.cpp")
mqt_copy_qdmi_runtime(imported-device-consumer ${device})
