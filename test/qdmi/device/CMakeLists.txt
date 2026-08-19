# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET MQT::CoreQDMI)
  return()
endif()

add_library(mqt-core-qdmi-session-device SHARED session_device.cpp)
target_link_libraries(mqt-core-qdmi-session-device PRIVATE qdmi::qdmi)
set_target_properties(
  mqt-core-qdmi-session-device
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/device"
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/device")

add_library(mqt-core-qdmi-metadata-device SHARED metadata_device.cpp)
set_target_properties(
  mqt-core-qdmi-metadata-device
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/device"
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/device"
             QDMI_DEVICE_ID "test.metadata-only"
             QDMI_DEVICE_PREFIX "TEST_METADATA"
             QDMI_RUNTIME_FILES "metadata-runtime.json")
add_custom_command(
  TARGET mqt-core-qdmi-metadata-device
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/metadata-runtime.json"
          "$<TARGET_FILE_DIR:mqt-core-qdmi-metadata-device>/metadata-runtime.json")
set_property(
  TARGET mqt-core-qdmi-metadata-device
  APPEND
  PROPERTY EXPORT_PROPERTIES QDMI_DEVICE_ID QDMI_DEVICE_PREFIX QDMI_RUNTIME_FILES)
set(metadata_device_export
    "${CMAKE_CURRENT_BINARY_DIR}/mqt-core-qdmi-metadata-device-targets.cmake")
export(TARGETS mqt-core-qdmi-metadata-device FILE "${metadata_device_export}")

set(imported_device_build_dir "${CMAKE_CURRENT_BINARY_DIR}/imported-device-consumer")
set(imported_device_configure_command
    ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_SOURCE_DIR}/imported_device" -B
    "${imported_device_build_dir}" -G "${CMAKE_GENERATOR}"
    "-DMQT_CORE_QDMI_DEVICE_TARGETS=${metadata_device_export}"
    "-DMQT_CORE_QDMI_HELPER=${PROJECT_SOURCE_DIR}/cmake/AddMQTQDMIDevice.cmake")
if(CMAKE_GENERATOR_PLATFORM)
  list(APPEND imported_device_configure_command -A "${CMAKE_GENERATOR_PLATFORM}")
endif()
if(CMAKE_GENERATOR_TOOLSET)
  list(APPEND imported_device_configure_command -T "${CMAKE_GENERATOR_TOOLSET}")
endif()
add_test(NAME mqt-core-qdmi-imported-device-configure COMMAND ${imported_device_configure_command})
add_test(NAME mqt-core-qdmi-imported-device-build
         COMMAND ${CMAKE_COMMAND} --build "${imported_device_build_dir}" --config $<CONFIG>)
set_tests_properties(mqt-core-qdmi-imported-device-configure
                     PROPERTIES FIXTURES_SETUP mqt-core-qdmi-imported-device)
set_tests_properties(mqt-core-qdmi-imported-device-build PROPERTIES FIXTURES_REQUIRED
                                                                    mqt-core-qdmi-imported-device)

set(runtime_file_device_build_dir "${CMAKE_CURRENT_BINARY_DIR}/runtime-file-device")
set(runtime_file_device_install_dir "${runtime_file_device_build_dir}/install")
if(WIN32)
  set(runtime_file_device_artifact_dir "${CMAKE_INSTALL_BINDIR}")
else()
  set(runtime_file_device_artifact_dir "${CMAKE_INSTALL_LIBDIR}")
endif()
set(runtime_file_device_configure_command
    ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_SOURCE_DIR}/runtime_file_device" -B
    "${runtime_file_device_build_dir}" -G "${CMAKE_GENERATOR}"
    "-DMQT_CORE_QDMI_HELPER=${PROJECT_SOURCE_DIR}/cmake/AddMQTQDMIDevice.cmake"
    "-DMQT_CORE_QDMI_TEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}" "-DCMAKE_BUILD_TYPE=$<CONFIG>")
if(CMAKE_GENERATOR_PLATFORM)
  list(APPEND runtime_file_device_configure_command -A "${CMAKE_GENERATOR_PLATFORM}")
endif()
if(CMAKE_GENERATOR_TOOLSET)
  list(APPEND runtime_file_device_configure_command -T "${CMAKE_GENERATOR_TOOLSET}")
endif()
add_test(NAME mqt-core-qdmi-runtime-file-configure COMMAND ${runtime_file_device_configure_command})
add_test(NAME mqt-core-qdmi-runtime-file-build
         COMMAND ${CMAKE_COMMAND} --build "${runtime_file_device_build_dir}" --config $<CONFIG>)
add_test(NAME mqt-core-qdmi-runtime-file-install
         COMMAND ${CMAKE_COMMAND} --install "${runtime_file_device_build_dir}" --config $<CONFIG>
                 --prefix "${runtime_file_device_install_dir}")
add_test(
  NAME mqt-core-qdmi-runtime-file-install-verify
  COMMAND
    ${CMAKE_COMMAND} -E compare_files
    "${runtime_file_device_install_dir}/${runtime_file_device_artifact_dir}/metadata-runtime.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/metadata-runtime.json")
set_tests_properties(mqt-core-qdmi-runtime-file-configure
                     PROPERTIES FIXTURES_SETUP mqt-core-qdmi-runtime-file-configure)
set_tests_properties(
  mqt-core-qdmi-runtime-file-build
  PROPERTIES FIXTURES_REQUIRED mqt-core-qdmi-runtime-file-configure FIXTURES_SETUP
             mqt-core-qdmi-runtime-file-build)
set_tests_properties(
  mqt-core-qdmi-runtime-file-install PROPERTIES FIXTURES_REQUIRED mqt-core-qdmi-runtime-file-build
                                                FIXTURES_SETUP mqt-core-qdmi-runtime-file-install)
set_tests_properties(mqt-core-qdmi-runtime-file-install-verify
                     PROPERTIES FIXTURES_REQUIRED mqt-core-qdmi-runtime-file-install)

function(add_qdmi_configuration_rejection_test name invalid_configuration)
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/runtime-file-device-invalid-${name}")
  set(configure_command
      ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_SOURCE_DIR}/runtime_file_device" -B "${build_dir}" -G
      "${CMAKE_GENERATOR}"
      "-DMQT_CORE_QDMI_HELPER=${PROJECT_SOURCE_DIR}/cmake/AddMQTQDMIDevice.cmake"
      "-DMQT_CORE_QDMI_TEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DMQT_CORE_QDMI_INVALID_CONFIGURATION=${invalid_configuration}"
      "-DCMAKE_BUILD_TYPE=$<CONFIG>")
  if(CMAKE_GENERATOR_PLATFORM)
    list(APPEND configure_command -A "${CMAKE_GENERATOR_PLATFORM}")
  endif()
  if(CMAKE_GENERATOR_TOOLSET)
    list(APPEND configure_command -T "${CMAKE_GENERATOR_TOOLSET}")
  endif()
  add_test(NAME "mqt-core-qdmi-configuration-rejects-${name}" COMMAND ${configure_command})
  set_tests_properties("mqt-core-qdmi-configuration-rejects-${name}" PROPERTIES WILL_FAIL TRUE)
endfunction()

add_qdmi_configuration_rejection_test(empty-id empty-id)
add_qdmi_configuration_rejection_test(empty-file empty-file)
add_qdmi_configuration_rejection_test(duplicate-base-id duplicate-base-id)
add_qdmi_configuration_rejection_test(duplicate-configured-id duplicate-configured-id)

if(NOT BUILD_MQT_CORE_QDMI_DDSIM_DEVICE OR NOT BUILD_MQT_CORE_QDMI_SC_DEVICE)
  return()
endif()

generate_prefixed_qdmi_headers(TEST_LIFECYCLE)
add_library(mqt-core-qdmi-lifecycle-device SHARED lifecycle_device.cpp)
target_compile_definitions(mqt-core-qdmi-lifecycle-device
                           PRIVATE TEST_LIFECYCLE_QDMI_device_EXPORTS)
target_compile_features(mqt-core-qdmi-lifecycle-device PRIVATE cxx_std_20)
target_include_directories(mqt-core-qdmi-lifecycle-device
                           PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include")
target_link_libraries(mqt-core-qdmi-lifecycle-device PRIVATE MQT::ProjectOptions
                                                             MQT::ProjectWarnings)

set(TARGET_NAME mqt-core-qdmi-test)
package_add_test(${TARGET_NAME} MQT::CoreQDMI test_device.cpp test_device_api.cpp ../test_slurm.cpp)
target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src/qdmi
                                                  "${CMAKE_CURRENT_BINARY_DIR}/include")
target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQDMICommon qdmi::qdmi
                                             mqt-core-qdmi-lifecycle-device)
target_compile_definitions(
  ${TARGET_NAME}
  PRIVATE DDSIM_DEVICE_LIBRARY="$<TARGET_FILE:MQT::CoreQDMI_DDSIM_Device>"
          LIFECYCLE_DEVICE_LIBRARY="$<TARGET_FILE:mqt-core-qdmi-lifecycle-device>"
          MQT_CORE_QDMI_SLURM_TEST_DEVICE="$<TARGET_FILE:mqt-core-qdmi-session-device>"
          SC_DEVICE_LIBRARY="$<TARGET_FILE:MQT::CoreQDMIScDevice>")
add_dependencies(${TARGET_NAME} MQT::CoreQDMI_DDSIM_Device MQT::CoreQDMIScDevice
                 mqt-core-qdmi-session-device)
mqt_copy_qdmi_runtime(${TARGET_NAME})
mqt_copy_qdmi_runtime(${TARGET_NAME} mqt-core-qdmi-metadata-device)
