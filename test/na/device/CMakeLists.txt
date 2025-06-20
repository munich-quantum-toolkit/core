# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Set test target name
set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-na-device-test)
# Set QDMI prefix again for the tests
set(QDMI_PREFIX "MQT_NA")

if(TARGET qdmi::mqt_na_device)

  # ------------------------------------------------------------------------------
  # Non-Functional Tests
  # ------------------------------------------------------------------------------

  generate_device_defs_executable(${QDMI_PREFIX})

  # ------------------------------------------------------------------------------
  # Functional Tests
  # ------------------------------------------------------------------------------

  package_add_test(${TARGET_NAME} qdmi::mqt_na_device test_interface.cpp)
  add_dependencies(${TARGET_NAME} qdmi_test_mqt_na_device_defs)

  # set c++ standard
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

endif()
