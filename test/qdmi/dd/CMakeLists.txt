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
  package_add_test(${TARGET_NAME} MQT::CoreQDMI_DDSIM_Device test_device.cpp)
  # Add dependency to enforce implementation of all interface functions
  add_dependencies(${TARGET_NAME} qdmi_test_mqt_ddsim_device_defs)
  # Set the executable path and version information
  target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_VERSION="${MQT_CORE_VERSION}")
endif()
