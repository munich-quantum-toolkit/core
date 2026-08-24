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
  add_library(mqt-core-qdmi-incomplete-client SHARED client_runtime_driver.cpp)
  target_link_libraries(mqt-core-qdmi-incomplete-client PRIVATE qdmi::qdmi)
  target_compile_definitions(mqt-core-qdmi-incomplete-client
                             PRIVATE QDMI_driver_EXPORTS TEST_CLIENT_ABI=QDMI_CLIENT_ABI_VERSION)
  add_library(mqt-core-qdmi-incompatible-client SHARED client_runtime_driver.cpp)
  target_link_libraries(mqt-core-qdmi-incompatible-client PRIVATE qdmi::qdmi)
  target_compile_definitions(mqt-core-qdmi-incompatible-client
                             PRIVATE QDMI_driver_EXPORTS TEST_CLIENT_ABI=QDMI_MAKE_VERSION\(1,3,0\))
  add_library(mqt-core-qdmi-fake-client SHARED client_runtime_driver.cpp)
  target_link_libraries(mqt-core-qdmi-fake-client PRIVATE qdmi::qdmi)
  target_compile_definitions(
    mqt-core-qdmi-fake-client PRIVATE QDMI_driver_EXPORTS TEST_CLIENT_ABI=QDMI_CLIENT_ABI_VERSION
                                      TEST_FULL_CLIENT)

  package_add_test(mqt-core-qdmi-client-runtime-test MQT::CoreQDMI test_client_runtime.cpp)
  target_compile_definitions(
    mqt-core-qdmi-client-runtime-test
    PRIVATE "MQT_CORE_QDMI_TEST_DRIVER=\"$<TARGET_FILE:mqt-core-qdmi-fake-client>\""
            "MQT_CORE_QDMI_INCOMPLETE_DRIVER=\"$<TARGET_FILE:mqt-core-qdmi-incomplete-client>\""
            "MQT_CORE_QDMI_INCOMPATIBLE_DRIVER=\"$<TARGET_FILE:mqt-core-qdmi-incompatible-client>\""
  )
  add_dependencies(mqt-core-qdmi-client-runtime-test mqt-core-qdmi-incomplete-client
                   mqt-core-qdmi-incompatible-client mqt-core-qdmi-fake-client)
  mqt_copy_qdmi_runtime(mqt-core-qdmi-client-runtime-test)

  add_executable(mqt-core-qdmi-packaged-runtime-test test_packaged_runtime.cpp)
  target_link_libraries(mqt-core-qdmi-packaged-runtime-test PRIVATE MQT::CoreQDMI
                                                                    MQT::ProjectWarnings)
  target_compile_definitions(
    mqt-core-qdmi-packaged-runtime-test
    PRIVATE "MQT_CORE_QDMI_TEST_DRIVER_FILENAME=\"$<TARGET_FILE_NAME:MQT::CoreQDMIDriver>\"")
  mqt_copy_qdmi_runtime(mqt-core-qdmi-packaged-runtime-test)
  add_test(
    NAME mqt-core-qdmi-packaged-runtime
    COMMAND
      ${CMAKE_COMMAND} -E env --unset=MQT_CORE_QDMI_DRIVER -- ${CMAKE_COMMAND} -E chdir
      "$<TARGET_FILE_DIR:mqt-core-qdmi-packaged-runtime-test>"
      "./$<TARGET_FILE_NAME:mqt-core-qdmi-packaged-runtime-test>")

  if(WIN32)
    # Discover tests after the Driver and device runtime copies finish.
    set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)
  endif()
  package_add_test(${TARGET_NAME} MQT::CoreQDMI test_client.cpp test_slurm.cpp)
  if(WIN32)
    unset(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE)
  endif()
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
