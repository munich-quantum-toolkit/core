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
set(configurations "test.runtime-file.variant|metadata-runtime.json")
set(devices "test.runtime-file.session|base-url=https://device.example|custom3=session")
if(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "empty-id")
  set(configurations "|metadata-runtime.json")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "empty-file")
  set(configurations "test.runtime-file.variant|")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "duplicate-base-id")
  set(configurations "test.runtime-file|metadata-runtime.json")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "duplicate-configured-id")
  set(configurations "test.runtime-file.variant|metadata-runtime.json"
                     "test.runtime-file.variant|metadata-runtime.json")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "missing-session-parameter")
  set(devices "test.runtime-file.session")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "malformed-session-parameter")
  set(devices "test.runtime-file.session|custom3")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "empty-session-value")
  set(devices "test.runtime-file.session|custom3=")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "unknown-session-parameter")
  set(devices "test.runtime-file.session|custom6=session")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "credential-session-parameter")
  set(devices "test.runtime-file.session|token=secret")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "repeated-session-parameter")
  set(devices "test.runtime-file.session|custom3=session|custom3=session")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "unknown-session-runtime-file")
  set(devices "test.runtime-file.session|device-config-file=absent.json")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "reserved-session-parameter")
  set(devices "test.runtime-file.session|device-config-file=metadata-runtime.json|custom2=session")
elseif(MQT_CORE_QDMI_INVALID_CONFIGURATION STREQUAL "duplicate-session-id")
  set(devices "test.runtime-file.variant|custom3=session")
endif()
mqt_configure_qdmi_device(
  mqt-core-qdmi-runtime-file-device
  ID
  test.runtime-file
  PREFIX
  TEST_RUNTIME
  RUNTIME_FILES
  "${MQT_CORE_QDMI_TEST_SOURCE_DIR}/metadata-runtime.json"
  CONFIGURATIONS
  ${configurations}
  DEVICES
  ${devices})
