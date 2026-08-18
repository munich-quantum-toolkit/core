# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-qdmi-registry-test)

if(TARGET MQT::CoreQDMI)
  package_add_test(${TARGET_NAME} MQT::CoreQDMI test_device_registry.cpp)
  target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/test)
  if(TARGET MQT::CoreQDMI_DDSIM_Device)
    target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_QDMI_TEST_HAS_DDSIM)
  endif()
  if(TARGET MQT::CoreQDMIScDevice)
    target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_QDMI_TEST_HAS_SC)
  endif()
  mqt_get_qdmi_device_targets(QDMI_DEVICE_TARGETS)
  if(QDMI_DEVICE_TARGETS)
    mqt_copy_qdmi_runtime(${TARGET_NAME} ${QDMI_DEVICE_TARGETS})
  endif()
endif()
