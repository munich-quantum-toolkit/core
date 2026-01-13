# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreQDMIScDeviceGen
   AND TARGET MQT::CoreQDMIScDeviceGenerator
   AND TARGET MQT::CoreQDMIScDevice)
  # Set QDMI prefix again for the tests
  set(QDMI_PREFIX "MQT_SC")
  generate_device_defs_executable(${QDMI_PREFIX})
  # Set test target name
  set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-sc-device-test)
  # Add the test executable
  package_add_test(${TARGET_NAME} MQT::CoreQDMIScDevice test_generator.cpp test_app.cpp
                   test_device.cpp)
  # Add dependency to enforce implementation of all interface functions
  add_dependencies(${TARGET_NAME} qdmi_test_mqt_sc_device_defs)
  # Link further libraries
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQDMIScDeviceGen
                                               nlohmann_json::nlohmann_json)
  # Set the executable path and version information
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE EXECUTABLE_PATH="$<TARGET_FILE:MQT::CoreQDMIScDeviceGenerator>"
            MQT_CORE_VERSION="${MQT_CORE_VERSION}"
            SC_DEVICE_JSON="${PROJECT_SOURCE_DIR}/json/sc/device.json")
endif()
