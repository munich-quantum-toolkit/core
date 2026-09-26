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
  add_subdirectory("${qdmi_SOURCE_DIR}/examples/driver" qdmi-example-driver)
  foreach(kind IN ITEMS incomplete incompatible)
    add_library(mqt-core-qdmi-${kind}-client SHARED client_abi.cpp)
    target_link_libraries(mqt-core-qdmi-${kind}-client PRIVATE qdmi::qdmi)
    target_compile_definitions(mqt-core-qdmi-${kind}-client PRIVATE QDMI_driver_EXPORTS)
  endforeach()
  target_compile_definitions(mqt-core-qdmi-incompatible-client PRIVATE TEST_INCOMPATIBLE_DRIVER)

  package_add_test_with_working_dir(mqt-core-qdmi-client-runtime-test MQT::CoreQDMI
                                    "${CMAKE_CURRENT_BINARY_DIR}" test_client_runtime.cpp)
  target_link_libraries(mqt-core-qdmi-client-runtime-test PRIVATE nlohmann_json::nlohmann_json)
  target_compile_definitions(
    mqt-core-qdmi-client-runtime-test
    PRIVATE "MQT_CORE_QDMI_TEST_DRIVER=\"$<TARGET_FILE:qdmi::example_driver>\""
            "MQT_CORE_QDMI_INCOMPLETE_DRIVER=\"$<TARGET_FILE:mqt-core-qdmi-incomplete-client>\""
            "MQT_CORE_QDMI_INCOMPATIBLE_DRIVER=\"$<TARGET_FILE:mqt-core-qdmi-incompatible-client>\""
            "MQT_CORE_QDMI_SESSION_DEVICE=\"$<TARGET_FILE:mqt-core-qdmi-session-device>\""
            "MQT_CORE_QDMI_EXAMPLE_CONFIG=\"${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/qdmi.conf\"")
  file(
    GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/qdmi.conf"
    CONTENT "$<TARGET_FILE:mqt-core-qdmi-session-device> TEST_SESSION test.example\n")
  add_dependencies(
    mqt-core-qdmi-client-runtime-test qdmi_example_driver mqt-core-qdmi-incomplete-client
    mqt-core-qdmi-incompatible-client mqt-core-qdmi-session-device)
  mqt_copy_qdmi_runtime(mqt-core-qdmi-client-runtime-test)

  package_add_test(${TARGET_NAME} MQT::CoreQDMI test_client.cpp test_slurm.cpp)
  set(client_test_config "$<TARGET_FILE_DIR:${TARGET_NAME}>/client-test-catalog.json")
  file(
    GENERATE
    OUTPUT "${client_test_config}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"test.unsupported-program-formats\", \"library\": \"$<TARGET_FILE_NAME:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\"},\n      {\"id\": \"test.slurm.idle\", \"library\": \"$<TARGET_FILE_NAME:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\", \"session\": {\"custom4\": \"idle\"}},\n      {\"id\": \"test.slurm.busy\", \"library\": \"$<TARGET_FILE_NAME:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\", \"session\": {\"custom4\": \"busy\"}},\n      {\"id\": \"test.slurm.grammar\", \"library\": \"$<TARGET_FILE_NAME:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\", \"session\": {\"custom4\": \"idle\"}},\n      {\"id\": \"test.slurm.single\", \"library\": \"$<TARGET_FILE_NAME:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\", \"session\": {\"custom4\": \"idle\"}},\n      {\"id\": \"test.slurm.offline\", \"library\": \"$<TARGET_FILE_NAME:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\", \"session\": {\"custom4\": \"offline\"}},\n      {\"id\": \"test.slurm.error\", \"library\": \"$<TARGET_FILE_NAME:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\", \"session\": {\"custom4\": \"error\"}},\n      {\"id\": \"test.slurm.maintenance\", \"library\": \"$<TARGET_FILE_NAME:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\", \"session\": {\"custom4\": \"maintenance\"}},\n      {\"id\": \"test.slurm.calibration\", \"library\": \"$<TARGET_FILE_NAME:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\", \"session\": {\"custom4\": \"calibration\"}},\n      {\"id\": \"test.slurm.max\", \"library\": \"$<TARGET_FILE_NAME:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\", \"session\": {\"custom4\": \"max\"}}\n    ]\n  }\n}\n"
  )
  add_custom_command(
    TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:mqt-core-qdmi-session-device>"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>")
  target_compile_definitions(${TARGET_NAME}
                             PRIVATE "MQT_CORE_QDMI_CLIENT_TEST_CONFIG=\"${client_test_config}\"")
  if(TARGET MQT::CoreQDMI_DDSIM_Device)
    target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_QDMI_HAS_DDSIM_DEVICE)
  endif()
  add_dependencies(${TARGET_NAME} mqt-core-qdmi-session-device)
  mqt_copy_qdmi_runtime(${TARGET_NAME})
endif()
