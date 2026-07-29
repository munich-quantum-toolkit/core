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
get_target_property(device_id ${device} QDMI_DEVICE_ID)
get_target_property(device_prefix ${device} QDMI_DEVICE_PREFIX)
get_target_property(runtime_files ${device} QDMI_RUNTIME_FILES)
if(NOT device_id STREQUAL "test.metadata-only"
   OR NOT device_prefix STREQUAL "TEST_METADATA"
   OR NOT runtime_files STREQUAL "metadata-runtime.json")
  message(FATAL_ERROR "Exported QDMI device metadata was not preserved")
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp" "int main() { return 0; }\n")
add_executable(imported-device-consumer "${CMAKE_CURRENT_BINARY_DIR}/main.cpp")
mqt_copy_qdmi_runtime(imported-device-consumer ${device})
add_custom_command(
  TARGET imported-device-consumer
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E compare_files
    "$<TARGET_FILE_DIR:imported-device-consumer>/metadata-runtime.json"
    "$<TARGET_FILE_DIR:${device}>/metadata-runtime.json")
