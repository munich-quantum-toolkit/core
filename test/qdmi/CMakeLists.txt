# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(devices)
add_subdirectory(driver)
add_subdirectory(registry)

set(TARGET_NAME mqt-core-qdmi-test)

if(TARGET MQT::CoreQDMI)
  package_add_test(${TARGET_NAME} MQT::CoreQDMI test_client.cpp test_slurm.cpp)
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE "MQT_CORE_QDMI_SLURM_TEST_DEVICE=\"$<TARGET_FILE:mqt-core-qdmi-session-device>\"")
  if(TARGET MQT::CoreQDMI_DDSIM_Device)
    target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_QDMI_HAS_DDSIM_DEVICE)
  endif()
  add_dependencies(${TARGET_NAME} mqt-core-qdmi-session-device)
  mqt_copy_qdmi_runtime(${TARGET_NAME})
  if(TARGET mqt-core-qdmi-check)
    add_test(
      NAME mqt-core-qdmi-check-cli
      COMMAND
        ${CMAKE_COMMAND} "-DCHECKER=$<TARGET_FILE:mqt-core-qdmi-check>"
        "-DSESSION_DEVICE=$<TARGET_FILE:mqt-core-qdmi-session-device>"
        "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/checker" -P
        "${CMAKE_CURRENT_SOURCE_DIR}/test_check_cli.cmake")
  endif()
endif()
