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
  add_library(mqt-core-qdmi-targeted-failure-client SHARED client_runtime_driver.cpp)
  target_link_libraries(mqt-core-qdmi-targeted-failure-client PRIVATE qdmi::qdmi)
  target_compile_definitions(
    mqt-core-qdmi-targeted-failure-client
    PRIVATE QDMI_driver_EXPORTS TEST_CLIENT_ABI=QDMI_CLIENT_ABI_VERSION TEST_FULL_CLIENT
            TEST_TARGETED_EXTENSION)
  add_library(mqt-core-qdmi-targeted-init-failure-client SHARED client_runtime_driver.cpp)
  target_link_libraries(mqt-core-qdmi-targeted-init-failure-client PRIVATE qdmi::qdmi)
  target_compile_definitions(
    mqt-core-qdmi-targeted-init-failure-client
    PRIVATE QDMI_driver_EXPORTS TEST_CLIENT_ABI=QDMI_CLIENT_ABI_VERSION TEST_FULL_CLIENT
            TEST_TARGETED_INIT_FAILURE_EXTENSION)

  set(runtime_extension_manifest
      "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/client-runtime/runtime-extension-package.qdmi.json")
  file(
    GENERATE
    OUTPUT "${runtime_extension_manifest}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"package.runtime\", \"library\": \"$<TARGET_FILE:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\"}\n    ]\n  }\n}\n"
  )

  if(WIN32)
    # Discover tests after the Driver and device runtime copies finish.
    set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)
  endif()
  package_add_test(mqt-core-qdmi-client-runtime-test MQT::CoreQDMI test_client_runtime.cpp)
  if(WIN32)
    unset(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE)
  endif()
  target_compile_definitions(
    mqt-core-qdmi-client-runtime-test
    PRIVATE
      "MQT_CORE_QDMI_TEST_DRIVER=\"$<TARGET_FILE:mqt-core-qdmi-fake-client>\""
      "MQT_CORE_QDMI_INCOMPLETE_DRIVER=\"$<TARGET_FILE:mqt-core-qdmi-incomplete-client>\""
      "MQT_CORE_QDMI_INCOMPATIBLE_DRIVER=\"$<TARGET_FILE:mqt-core-qdmi-incompatible-client>\""
      "MQT_CORE_QDMI_TARGETED_FAILURE_DRIVER=\"$<TARGET_FILE:mqt-core-qdmi-targeted-failure-client>\""
      "MQT_CORE_QDMI_RUNTIME_EXTENSION_MANIFEST=\"${runtime_extension_manifest}\"")
  add_dependencies(
    mqt-core-qdmi-client-runtime-test mqt-core-qdmi-incomplete-client
    mqt-core-qdmi-incompatible-client mqt-core-qdmi-fake-client
    mqt-core-qdmi-targeted-failure-client mqt-core-qdmi-session-device)
  mqt_copy_qdmi_runtime(mqt-core-qdmi-client-runtime-test)

  if(WIN32)
    set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)
  endif()
  package_add_test(mqt-core-qdmi-targeted-selection-test MQT::CoreQDMI
                   test_client_targeted_selection.cpp)
  if(WIN32)
    unset(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE)
  endif()
  target_compile_definitions(
    mqt-core-qdmi-targeted-selection-test
    PRIVATE
      "MQT_CORE_QDMI_TEST_DRIVER=\"$<TARGET_FILE:mqt-core-qdmi-fake-client>\""
      "MQT_CORE_QDMI_TARGETED_INIT_FAILURE_DRIVER=\"$<TARGET_FILE:mqt-core-qdmi-targeted-init-failure-client>\""
  )
  add_dependencies(mqt-core-qdmi-targeted-selection-test mqt-core-qdmi-fake-client
                   mqt-core-qdmi-targeted-init-failure-client)
  mqt_copy_qdmi_runtime(mqt-core-qdmi-targeted-selection-test)

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

  set(extension_fixture_dir "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/default-driver-extension")
  set(extension_manifest "${extension_fixture_dir}/package-session.qdmi.json")
  set(unicode_extension_manifest_source
      "${extension_fixture_dir}/unicode-package-session.qdmi.json")
  set(late_extension_manifest "${extension_fixture_dir}/late-package-session.qdmi.json")
  set(post_freeze_extension_manifest
      "${extension_fixture_dir}/post-freeze-package-session.qdmi.json")
  set(conflicting_extension_manifest
      "${extension_fixture_dir}/conflicting-package-session.qdmi.json")
  set(incompatible_extension_manifest
      "${extension_fixture_dir}/incompatible-package-session.qdmi.json")
  set(malformed_extension_manifest "${extension_fixture_dir}/malformed-package-session.qdmi.json")
  set(missing_library_extension_manifest
      "${extension_fixture_dir}/missing-library-package-session.qdmi.json")
  set(missing_prefix_extension_manifest
      "${extension_fixture_dir}/missing-prefix-package-session.qdmi.json")
  set(nul_extension_manifest "${extension_fixture_dir}/nul-package-session.qdmi.json")
  set(conflicting_config_manifest
      "${extension_fixture_dir}/conflicting-config-package-session.qdmi.json")
  file(
    GENERATE
    OUTPUT "${extension_manifest}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"package.session\", \"library\": \"$<TARGET_FILE:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\", \"session\": {\"custom4\": \"idle\"}},\n      {\"id\": \"package.child-error\", \"library\": \"$<TARGET_FILE:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\", \"session\": {\"custom5\": \"permission-denied\"}}\n    ]\n  }\n}\n"
  )
  file(
    GENERATE
    OUTPUT "${unicode_extension_manifest_source}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"package.unicode\", \"library\": \"$<TARGET_FILE:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\"}\n    ]\n  }\n}\n"
  )
  file(
    GENERATE
    OUTPUT "${late_extension_manifest}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"package.late\", \"library\": \"$<TARGET_FILE:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\"}\n    ]\n  }\n}\n"
  )
  file(
    GENERATE
    OUTPUT "${post_freeze_extension_manifest}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"package.post-freeze\", \"library\": \"$<TARGET_FILE:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\"}\n    ]\n  }\n}\n"
  )
  file(
    GENERATE
    OUTPUT "${conflicting_extension_manifest}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"package.session\", \"library\": \"$<TARGET_FILE:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\"}\n    ]\n  }\n}\n"
  )
  file(
    GENERATE
    OUTPUT "${incompatible_extension_manifest}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"package.incompatible\", \"library\": \"$<TARGET_FILE:mqt-core-qdmi-incomplete-client>\", \"prefix\": \"MISSING\"}\n    ]\n  }\n}\n"
  )
  file(
    GENERATE
    OUTPUT "${malformed_extension_manifest}"
    CONTENT "{\n")
  file(
    GENERATE
    OUTPUT "${missing_library_extension_manifest}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"package.missing-library\", \"library\": \"missing-device-library\", \"prefix\": \"MISSING\"}\n    ]\n  }\n}\n"
  )
  file(
    GENERATE
    OUTPUT "${missing_prefix_extension_manifest}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"package.missing-prefix\", \"library\": \"$<TARGET_FILE:mqt-core-qdmi-session-device>\"}\n    ]\n  }\n}\n"
  )
  file(
    GENERATE
    OUTPUT "${nul_extension_manifest}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"package.nul\", \"library\": \"$<TARGET_FILE:mqt-core-qdmi-session-device>\\u0000alias\", \"prefix\": \"TEST_SESSION\"}\n    ]\n  }\n}\n"
  )
  file(
    GENERATE
    OUTPUT "${conflicting_config_manifest}"
    CONTENT
      "{\n  \"schema-version\": 1,\n  \"qdmi\": {\n    \"devices\": [\n      {\"id\": \"package.bad-config\", \"library\": \"$<TARGET_FILE:mqt-core-qdmi-session-device>\", \"prefix\": \"TEST_SESSION\", \"session\": {\"device-config\": {\"inline\": {}}, \"custom1\": \"raw\"}}\n    ]\n  }\n}\n"
  )
  if(WIN32)
    set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)
  endif()
  package_add_test(mqt-core-qdmi-default-driver-extension-test MQT::CoreQDMI
                   test_default_driver_extension.cpp)
  if(WIN32)
    unset(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE)
  endif()
  target_compile_definitions(
    mqt-core-qdmi-default-driver-extension-test
    PRIVATE "MQT_CORE_QDMI_FAKE_CLIENT=\"$<TARGET_FILE:mqt-core-qdmi-fake-client>\""
            "MQT_CORE_QDMI_PACKAGE_MANIFEST=\"${extension_manifest}\""
            "MQT_CORE_QDMI_UNICODE_MANIFEST_SOURCE=\"${unicode_extension_manifest_source}\""
            "MQT_CORE_QDMI_LATE_MANIFEST=\"${late_extension_manifest}\""
            "MQT_CORE_QDMI_POST_FREEZE_MANIFEST=\"${post_freeze_extension_manifest}\""
            "MQT_CORE_QDMI_CONFLICTING_MANIFEST=\"${conflicting_extension_manifest}\""
            "MQT_CORE_QDMI_INCOMPATIBLE_MANIFEST=\"${incompatible_extension_manifest}\""
            "MQT_CORE_QDMI_MALFORMED_MANIFEST=\"${malformed_extension_manifest}\""
            "MQT_CORE_QDMI_MISSING_LIBRARY_MANIFEST=\"${missing_library_extension_manifest}\""
            "MQT_CORE_QDMI_MISSING_PREFIX_MANIFEST=\"${missing_prefix_extension_manifest}\""
            "MQT_CORE_QDMI_NUL_MANIFEST=\"${nul_extension_manifest}\""
            "MQT_CORE_QDMI_CONFLICTING_CONFIG_MANIFEST=\"${conflicting_config_manifest}\"")
  add_dependencies(
    mqt-core-qdmi-default-driver-extension-test mqt-core-qdmi-session-device
    mqt-core-qdmi-incomplete-client mqt-core-qdmi-fake-client ${MQT_CORE_TARGET_NAME}-qdmi-driver)
  mqt_copy_qdmi_runtime(mqt-core-qdmi-default-driver-extension-test)

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
