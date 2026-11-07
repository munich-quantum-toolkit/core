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
    add_subdirectory("${qdmi_SOURCE_DIR}/examples/driver" qdmi-example-driver)
    foreach(kind IN ITEMS incomplete incompatible)
      add_library(mqt-core-qdmi-${kind}-client SHARED client_abi.cpp)
      target_link_libraries(mqt-core-qdmi-${kind}-client PRIVATE qdmi::qdmi)
      target_compile_definitions(mqt-core-qdmi-${kind}-client PRIVATE QDMI_driver_EXPORTS)
    endforeach()
    target_compile_definitions(mqt-core-qdmi-incompatible-client PRIVATE TEST_INCOMPATIBLE_DRIVER)
    # Keep generated configuration below the example driver's allowed working directory.
    package_add_test_with_working_dir(${TARGET_NAME} MQT::CoreQDMI "${CMAKE_CURRENT_BINARY_DIR}"
                                      test_client.cpp test_slurm.cpp)
    target_compile_definitions(
      ${TARGET_NAME}
      PRIVATE "MQT_CORE_QDMI_SLURM_TEST_DEVICE=\"$<TARGET_FILE:mqt-core-qdmi-session-device>\"")
    add_dependencies(${TARGET_NAME} mqt-core-qdmi-session-device)
    target_compile_definitions(
      ${TARGET_NAME}
      PRIVATE
        "MQT_CORE_QDMI_TEST_DRIVER=\"$<TARGET_FILE:qdmi::example_driver>\""
        "MQT_CORE_QDMI_INCOMPLETE_DRIVER=\"$<TARGET_FILE:mqt-core-qdmi-incomplete-client>\""
        "MQT_CORE_QDMI_INCOMPATIBLE_DRIVER=\"$<TARGET_FILE:mqt-core-qdmi-incompatible-client>\""
        "MQT_CORE_QDMI_EXAMPLE_CONFIG=\"${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/qdmi.conf\"")
    file(
      GENERATE
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/qdmi.conf"
      CONTENT
        "$<TARGET_FILE:mqt-core-qdmi-session-device> TEST_SESSION test.example\n$<TARGET_FILE:MQT::CoreQDMI_DDSIM_Device> MQT_DDSIM test.external-ddsim\n")
    add_dependencies(${TARGET_NAME} qdmi_example_driver mqt-core-qdmi-incomplete-client
                     mqt-core-qdmi-incompatible-client)
    mqt_copy_qdmi_runtime(${TARGET_NAME})
  endif()
endif()
add_subdirectory(registry)
