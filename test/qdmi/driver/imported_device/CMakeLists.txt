# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.24)
project(mqt-core-qdmi-imported-device-test LANGUAGES CXX)

if(NOT MQT_CORE_BUILD_CONFIG
   OR NOT MQT_CORE_BUILD_DIR
   OR NOT MQT_CORE_INSTALL_DIR
   OR NOT MQT_CORE_QDMI_DEVICE_TARGETS)
  message(FATAL_ERROR "The MQT Core build, install prefix, and device target export are required")
endif()

file(REMOVE_RECURSE "${MQT_CORE_INSTALL_DIR}")
foreach(component IN ITEMS qdmi_Runtime qdmi_Development mqt-core_Runtime mqt-core_Development)
  execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${MQT_CORE_BUILD_DIR}" --config "${MQT_CORE_BUILD_CONFIG}"
            --prefix "${MQT_CORE_INSTALL_DIR}" --component "${component}"
    RESULT_VARIABLE install_result)
  if(install_result)
    message(FATAL_ERROR "Failed to stage installed component ${component}")
  endif()
endforeach()

list(PREPEND CMAKE_PREFIX_PATH "${MQT_CORE_INSTALL_DIR}")
find_package(mqt-core CONFIG REQUIRED PATHS "${MQT_CORE_INSTALL_DIR}" NO_DEFAULT_PATH)
include("${MQT_CORE_QDMI_DEVICE_TARGETS}")

get_target_property(client_imported MQT::CoreQDMI IMPORTED)
get_target_property(driver_imported MQT::CoreQDMIDriver IMPORTED)
if(NOT client_imported OR NOT driver_imported)
  message(FATAL_ERROR "The installed QDMI Client and Driver targets must be imported")
endif()

set(device mqt-core-qdmi-metadata-device)
if(WIN32)
  set(client_runtime_dependency "${CMAKE_CURRENT_BINARY_DIR}/client-runtime-dependency.dll")
  set(client_runtime_import_library "${CMAKE_CURRENT_BINARY_DIR}/client-runtime-dependency.lib")
  file(WRITE "${client_runtime_dependency}" "client runtime dependency\n")
  file(WRITE "${client_runtime_import_library}" "")
  add_library(qdmi-client-runtime-dependency SHARED IMPORTED)
  set_target_properties(
    qdmi-client-runtime-dependency PROPERTIES IMPORTED_IMPLIB "${client_runtime_import_library}"
                                              IMPORTED_LOCATION "${client_runtime_dependency}")
  set_property(
    TARGET MQT::CoreQDMI
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES qdmi-client-runtime-dependency)

  set(driver_runtime_dependency "${CMAKE_CURRENT_BINARY_DIR}/driver-runtime-dependency.dll")
  set(driver_runtime_import_library "${CMAKE_CURRENT_BINARY_DIR}/driver-runtime-dependency.lib")
  file(WRITE "${driver_runtime_dependency}" "driver runtime dependency\n")
  file(WRITE "${driver_runtime_import_library}" "")
  add_library(qdmi-driver-runtime-dependency SHARED IMPORTED)
  set_target_properties(
    qdmi-driver-runtime-dependency PROPERTIES IMPORTED_IMPLIB "${driver_runtime_import_library}"
                                              IMPORTED_LOCATION "${driver_runtime_dependency}")
  set_property(
    TARGET MQT::CoreQDMIDriver
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES qdmi-driver-runtime-dependency)

  set(device_runtime_dependency "${CMAKE_CURRENT_BINARY_DIR}/device-runtime-dependency.dll")
  set(device_runtime_import_library "${CMAKE_CURRENT_BINARY_DIR}/device-runtime-dependency.lib")
  file(WRITE "${device_runtime_dependency}" "device runtime dependency\n")
  file(WRITE "${device_runtime_import_library}" "")
  add_library(qdmi-device-runtime-dependency SHARED IMPORTED)
  set_target_properties(
    qdmi-device-runtime-dependency PROPERTIES IMPORTED_IMPLIB "${device_runtime_import_library}"
                                              IMPORTED_LOCATION "${device_runtime_dependency}")
  set_property(
    TARGET ${device}
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES qdmi-device-runtime-dependency)
endif()

get_target_property(device_id ${device} QDMI_DEVICE_ID)
get_target_property(device_prefix ${device} QDMI_DEVICE_PREFIX)
get_target_property(runtime_files ${device} QDMI_RUNTIME_FILES)
if(NOT device_id STREQUAL "test.metadata-only"
   OR NOT device_prefix STREQUAL "TEST_METADATA"
   OR NOT runtime_files STREQUAL "metadata-runtime.json")
  message(FATAL_ERROR "Exported QDMI device metadata was not preserved")
endif()

file(
  WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp"
  [=[#ifdef _WIN32
#include <windows.h>

int main(int argc, char** argv) {
  if (argc != 3) {
    return 1;
  }
  const auto client = LoadLibraryA(argv[1]);
  const auto driver = LoadLibraryA(argv[2]);
  if (driver != nullptr) {
    FreeLibrary(driver);
  }
  if (client != nullptr) {
    FreeLibrary(client);
  }
  return client != nullptr && driver != nullptr ? 0 : 1;
}
#else
#include "qdmi/Client.hpp"

int main() {
  const qdmi::Session session;
  return 0;
}
#endif
]=])
add_executable(imported-device-consumer "${CMAKE_CURRENT_BINARY_DIR}/main.cpp")
if(NOT WIN32)
  target_link_libraries(imported-device-consumer PRIVATE MQT::CoreQDMI)
endif()
set_property(TARGET imported-device-consumer PROPERTY BUILD_WITH_INSTALL_RPATH TRUE)
mqt_copy_qdmi_runtime(imported-device-consumer ${device})
get_target_property(build_with_install_rpath imported-device-consumer BUILD_WITH_INSTALL_RPATH)
if(build_with_install_rpath)
  message(FATAL_ERROR "QDMI runtime staging must use the build RPATH")
endif()

string(MAKE_C_IDENTIFIER "imported-device-consumer-${device}" manifest_stem)
set(manifest_name "${manifest_stem}.qdmi.json")
add_custom_command(
  TARGET imported-device-consumer
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E compare_files
    "$<TARGET_FILE_DIR:imported-device-consumer>/$<TARGET_FILE_NAME:MQT::CoreQDMI>"
    "$<TARGET_FILE:MQT::CoreQDMI>"
  COMMAND
    ${CMAKE_COMMAND} -E compare_files
    "$<TARGET_FILE_DIR:imported-device-consumer>/$<TARGET_FILE_NAME:MQT::CoreQDMIDriver>"
    "$<TARGET_FILE:MQT::CoreQDMIDriver>"
  COMMAND
    ${CMAKE_COMMAND} -E compare_files
    "$<TARGET_FILE_DIR:imported-device-consumer>/$<TARGET_FILE_NAME:${device}>"
    "$<TARGET_FILE:${device}>"
  COMMAND
    ${CMAKE_COMMAND} -E compare_files
    "$<TARGET_FILE_DIR:imported-device-consumer>/metadata-runtime.json"
    "$<TARGET_FILE_DIR:${device}>/metadata-runtime.json"
  COMMAND
    ${CMAKE_COMMAND} -E compare_files
    "$<TARGET_FILE_DIR:imported-device-consumer>/${manifest_name}"
    "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${manifest_name}")

if(WIN32)
  add_custom_command(
    TARGET imported-device-consumer
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E compare_files
      "$<TARGET_FILE_DIR:imported-device-consumer>/$<TARGET_FILE_NAME:qdmi-client-runtime-dependency>"
      "$<TARGET_FILE:qdmi-client-runtime-dependency>"
    COMMAND
      ${CMAKE_COMMAND} -E compare_files
      "$<TARGET_FILE_DIR:imported-device-consumer>/$<TARGET_FILE_NAME:qdmi-driver-runtime-dependency>"
      "$<TARGET_FILE:qdmi-driver-runtime-dependency>"
    COMMAND
      ${CMAKE_COMMAND} -E compare_files
      "$<TARGET_FILE_DIR:imported-device-consumer>/$<TARGET_FILE_NAME:qdmi-device-runtime-dependency>"
      "$<TARGET_FILE:qdmi-device-runtime-dependency>")
endif()

add_custom_command(
  TARGET imported-device-consumer
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E env
    "MQT_CORE_QDMI_DRIVER=$<TARGET_FILE_DIR:imported-device-consumer>/$<TARGET_FILE_NAME:MQT::CoreQDMIDriver>"
    "$<TARGET_FILE:imported-device-consumer>"
    "$<TARGET_FILE_DIR:imported-device-consumer>/$<TARGET_FILE_NAME:MQT::CoreQDMI>"
    "$<TARGET_FILE_DIR:imported-device-consumer>/$<TARGET_FILE_NAME:MQT::CoreQDMIDriver>")
