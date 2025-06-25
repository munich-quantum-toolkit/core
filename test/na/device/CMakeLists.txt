# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreNaDeviceGen)
  # Set test target name
  set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-na-device-gen-test)
  package_add_test(${TARGET_NAME} MQT::CoreNaDeviceGen test_generator.cpp)
endif()

if(TARGET MQT::CoreQdmiNaDeviceGenerator)
  # Set test target name
  set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-na-device-generator-test)
  package_add_test(${TARGET_NAME} "" test_app.cpp)
  target_compile_definitions(
    ${TARGET_NAME} PRIVATE EXECUTABLE_PATH="$<TARGET_FILE:MQT::CoreQdmiNaDeviceGenerator>"
                           MQT_CORE_VERSION="${MQT_CORE_VERSION}")
endif()

if(TARGET qdmi::mqt_na_device)
  # Set test target name
  set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-na-device-test)
  # Set QDMI prefix again for the tests
  set(QDMI_PREFIX "MQT_NA")

  # ------------------------------------------------------------------------------
  # Non-Functional Tests
  # ------------------------------------------------------------------------------

  generate_device_defs_executable(${QDMI_PREFIX})

  # ------------------------------------------------------------------------------
  # Functional Tests
  # ------------------------------------------------------------------------------

  package_add_test(${TARGET_NAME} qdmi::mqt_na_device test_device.cpp)
  target_link_libraries(${TARGET_NAME} PRIVATE nlohmann_json::nlohmann_json)
  add_dependencies(${TARGET_NAME} qdmi_test_mqt_na_device_defs)
  target_compile_definitions(${TARGET_NAME}
                             PRIVATE NA_DEVICE_JSON="${CMAKE_SOURCE_DIR}/json/na/device.json")

  # set c++ standard
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

endif()
