# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.24)
project(mqt-core-qdmi-runtime-file-device-test LANGUAGES CXX)

if(NOT MQT_CORE_QDMI_HELPER OR NOT MQT_CORE_QDMI_TEST_SOURCE_DIR)
  message(FATAL_ERROR "The QDMI helper and test source directory are required")
endif()

set(QDMI_VERSION "test")
include("${MQT_CORE_QDMI_HELPER}")

add_library(mqt-core-qdmi-runtime-file-device SHARED
            "${MQT_CORE_QDMI_TEST_SOURCE_DIR}/metadata_device.cpp")
mqt_configure_qdmi_device(
  mqt-core-qdmi-runtime-file-device
  ID
  test.runtime-file
  PREFIX
  TEST_RUNTIME
  RUNTIME_FILES
  "${MQT_CORE_QDMI_TEST_SOURCE_DIR}/metadata-runtime.json")
