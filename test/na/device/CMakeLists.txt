# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreNaDeviceGen
   AND TARGET MQT::CoreQdmiNaDeviceGenerator
   AND TARGET qdmi::mqt_na_device)
  # Set QDMI prefix again for the tests
  set(QDMI_PREFIX "MQT_NA")
  generate_device_defs_executable(${QDMI_PREFIX})
  # Set test target name
  set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-na-device-test)
  # Add the test executable
  package_add_test(${TARGET_NAME} qdmi::mqt_na_device test_generator.cpp test_app.cpp
                   test_device.cpp)
  # Add dependency to enforce implementation of all interface functions
  add_dependencies(${TARGET_NAME} qdmi_test_mqt_na_device_defs)
  # Link further libraries
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreNaDeviceGen nlohmann_json::nlohmann_json)
  # Set the executable path and version information
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE EXECUTABLE_PATH="$<TARGET_FILE:MQT::CoreQdmiNaDeviceGenerator>"
            MQT_CORE_VERSION="${MQT_CORE_VERSION}"
            NA_DEVICE_JSON="${PROJECT_SOURCE_DIR}/json/na/device.json")
  # Set c++ standard
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)
endif()
