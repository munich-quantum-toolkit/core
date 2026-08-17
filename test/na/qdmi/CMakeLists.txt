# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreNAQDMI)
  # Set test target name
  set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-na-qdmi-test)
  # Add the test executable
  package_add_test(${TARGET_NAME} MQT::CoreNAQDMI test_qdmi.cpp)
  # Set the device json path
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE NA_DEVICE_JSON="${PROJECT_SOURCE_DIR}/json/na/mqt-core-qdmi-na-device.json")
  mqt_copy_qdmi_runtime(${TARGET_NAME})
endif()
