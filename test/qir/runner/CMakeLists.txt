# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreQIRRunner)
  # Set test target name
  set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qir-runner-test)
  # Add the test executable
  package_add_test(${TARGET_NAME} "" test_qir_runner.cpp)
  # Collect QIR files
  set(QIR_FILES "")
  set(QIR_FILES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
  file(GLOB QIR_FILES "${QIR_FILES_DIR}/*.ll")
  # transform QIR_EXECUTABLES to comma separated list of string ("...")
  string(REPLACE ";" "\",\"" QIR_FILES "\"${QIR_FILES}\"")
  # Set the executable path and version information
  target_compile_definitions(
    ${TARGET_NAME} PRIVATE EXECUTABLE_PATH="$<TARGET_FILE:MQT::CoreQIRRunner>"
                           QIR_FILES=${QIR_FILES})
  # Set c++ standard
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
endif()
