# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(devices)
if(BUILD_MQT_CORE_QDMI_DDSIM_DEVICE AND BUILD_MQT_CORE_QDMI_SC_DEVICE)
  add_subdirectory(driver)

  set(TARGET_NAME mqt-core-qdmi-test)

  if(TARGET MQT::CoreQDMI)
    package_add_test(${TARGET_NAME} MQT::CoreQDMI test_client.cpp test_slurm.cpp)
    target_compile_definitions(
      ${TARGET_NAME}
      PRIVATE "MQT_CORE_QDMI_SLURM_TEST_DEVICE=\"$<TARGET_FILE:mqt-core-qdmi-session-device>\"")
    add_dependencies(${TARGET_NAME} mqt-core-qdmi-session-device)
    mqt_copy_qdmi_runtime(${TARGET_NAME})
  endif()
endif()
add_subdirectory(registry)
