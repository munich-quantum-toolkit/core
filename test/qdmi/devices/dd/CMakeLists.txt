# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreQDMI_DDSIM_Device)
  # Set QDMI prefix again for the tests
  set(QDMI_PREFIX "MQT_DDSIM")
  generate_device_defs_executable(${QDMI_PREFIX})
  # Set test target name
  set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-ddsim-device-test)
  # Add the test executable
  package_add_test(
    ${TARGET_NAME}
    MQT::CoreQDMI_DDSIM_Device
    helpers/test_utils.cpp
    device_properties_test.cpp
    session_lifecycle_test.cpp
    job_parameters_test.cpp
    job_lifecycle_test.cpp
    results_sampling_test.cpp
    results_statevector_test.cpp
    results_probabilities_test.cpp
    error_handling_test.cpp
    concurrency_test.cpp
    device_status_test.cpp)
  # Add dependency to enforce implementation of all interface functions
  add_dependencies(${TARGET_NAME} qdmi_test_mqt_ddsim_device_defs)
  # Set the executable path and version information
  target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_VERSION="${MQT_CORE_VERSION}")
  target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
