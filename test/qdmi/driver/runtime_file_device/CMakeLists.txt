# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.28)
project(mqt-core-qdmi-runtime-file-device-test LANGUAGES CXX)

if(NOT MQT_CORE_QDMI_HELPER OR NOT MQT_CORE_QDMI_TEST_SOURCE_DIR)
  message(FATAL_ERROR "The QDMI helper and test source directory are required")
endif()

set(QDMI_VERSION "test")
set(MQT_CORE_INSTALL ON)
set(MQT_CORE_TARGET_NAME mqt-core)
include("${MQT_CORE_QDMI_HELPER}")
configure_file("${MQT_CORE_QDMI_TEST_SOURCE_DIR}/metadata-runtime.json"
               "${CMAKE_CURRENT_BINARY_DIR}/inputs/metadata-runtime.json" COPYONLY)

add_library(mqt-core-qdmi-runtime-file-device SHARED
            "${MQT_CORE_QDMI_TEST_SOURCE_DIR}/metadata_device.cpp")
set(configurations "test.runtime-file.variant|metadata-runtime.json")
if(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "empty-id")
  set(configurations "|metadata-runtime.json")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "empty-file")
  set(configurations "test.runtime-file.variant|")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "duplicate-base-id")
  set(configurations "test.runtime-file|metadata-runtime.json")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "duplicate-configured-id")
  set(configurations "test.runtime-file.variant|metadata-runtime.json"
                     "test.runtime-file.variant|metadata-runtime.json")
endif()
mqt_configure_qdmi_device(
  mqt-core-qdmi-runtime-file-device
  ID
  test.runtime-file
  PREFIX
  TEST_RUNTIME
  RUNTIME_FILES
  "${CMAKE_CURRENT_BINARY_DIR}/inputs/metadata-runtime.json"
  CONFIGURATIONS
  ${configurations})

add_library(runtime-consumer MODULE "${MQT_CORE_QDMI_TEST_SOURCE_DIR}/metadata_device.cpp")
set_target_properties(runtime-consumer PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                  "${CMAKE_CURRENT_BINARY_DIR}/consumer")
mqt_copy_qdmi_runtime(runtime-consumer mqt-core-qdmi-runtime-file-device)
file(
  GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/check-incremental-$<CONFIG>.cmake"
  CONTENT
    "set(build_dir \"${CMAKE_CURRENT_BINARY_DIR}\")\nset(config \"$<CONFIG>\")\nset(device \"$<TARGET_FILE:mqt-core-qdmi-runtime-file-device>\")\nset(consumer_dir \"$<TARGET_FILE_DIR:runtime-consumer>\")\ninclude(\"${CMAKE_CURRENT_SOURCE_DIR}/check_incremental.cmake\")\n"
)
