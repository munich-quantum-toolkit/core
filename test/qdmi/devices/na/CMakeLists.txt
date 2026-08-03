# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreQDMINaDeviceConfig AND TARGET MQT::CoreQDMINaDevice)
  # Set QDMI prefix again for the tests
  set(QDMI_PREFIX "MQT_NA")
  generate_device_defs_executable(${QDMI_PREFIX} TARGET MQT::CoreQDMINaDevice)
  # Set test target name
  set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-na-device-test)
  if(WIN32)
    # Discover tests after the provider and its runtime assets have been copied.
    set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)
  endif()
  # Add the test executable
  package_add_test(${TARGET_NAME} MQT::CoreQDMINaDevice test_configuration.cpp test_device.cpp)
  target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/test)
  # Add dependency to enforce implementation of all interface functions
  add_dependencies(${TARGET_NAME} qdmi_test_mqt_na_device_defs)
  # Link further libraries
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQDMINaDeviceConfig
                                               nlohmann_json::nlohmann_json)
  # Set the executable path and version information
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE MQT_CORE_VERSION="${MQT_CORE_VERSION}"
            NA_DEVICE_JSON="${PROJECT_SOURCE_DIR}/json/na/mqt-core-qdmi-na-device.json")

  # On Windows, copy the provider and its runtime assets beside the test executable.
  if(WIN32)
    mqt_copy_qdmi_runtime(${TARGET_NAME} MQT::CoreQDMINaDevice)
  endif()
endif()
