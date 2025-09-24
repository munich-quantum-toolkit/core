# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreNAFoMaC)
  # Set test target name
  set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-na-fomac-test)
  # Add the test executable
  package_add_test(${TARGET_NAME} MQT::CoreNAFoMaC test_device.cpp)
  # Link further libraries
  target_link_libraries(${TARGET_NAME} PRIVATE nlohmann_json::nlohmann_json)
  # Set the device json path
  target_compile_definitions(${TARGET_NAME}
                             PRIVATE NA_DEVICE_JSON="${PROJECT_SOURCE_DIR}/json/na/device.json")
endif()
