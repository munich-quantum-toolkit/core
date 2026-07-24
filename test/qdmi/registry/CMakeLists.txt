# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-qdmi-registry-test)

if(TARGET MQT::CoreQDMIDriver)
  package_add_test(${TARGET_NAME} MQT::CoreQDMIDriver test_device_registry.cpp)
  target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src/qdmi/driver)
  mqt_copy_qdmi_runtime(${TARGET_NAME})
endif()
