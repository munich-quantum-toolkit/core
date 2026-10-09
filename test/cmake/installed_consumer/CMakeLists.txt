# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.28)
project(mqt-core-installed-consumer LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)
set(CMAKE_C_COMPILER_LAUNCHER "${CMAKE_COMMAND};-E;env")
set(CMAKE_CXX_COMPILER_LAUNCHER "${CMAKE_COMMAND};-E;env")
set(DEPLOY OFF)
set(ENV{DEPLOY} ON)

set(consumer_settings
    CMAKE_BUILD_TYPE
    CMAKE_C_COMPILER_LAUNCHER
    CMAKE_CXX_COMPILER_LAUNCHER
    CMAKE_C_FLAGS
    CMAKE_CXX_FLAGS
    CMAKE_EXPORT_COMPILE_COMMANDS
    CMAKE_VERIFY_INTERFACE_HEADER_SETS
    CMAKE_INTERPROCEDURAL_OPTIMIZATION
    CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS
    CMAKE_CXX_SCAN_FOR_MODULES
    CMAKE_OSX_DEPLOYMENT_TARGET
    CMAKE_INSTALL_BINDIR
    CMAKE_INSTALL_LIBDIR
    CMAKE_INSTALL_DATADIR
    DEPLOY
    ENABLE_IPO
    ENABLE_CACHE)
foreach(setting IN LISTS consumer_settings)
  set(before_${setting} "${${setting}}")
  get_property(
    before_cache_${setting}
    CACHE ${setting}
    PROPERTY VALUE)
  get_property(
    before_cache_type_${setting}
    CACHE ${setting}
    PROPERTY TYPE)
  if(DEFINED ${setting})
    set(defined_${setting} TRUE)
  else()
    set(defined_${setting} FALSE)
  endif()
endforeach()
get_directory_property(before_compile_options COMPILE_OPTIONS)
set(consumer_properties USE_FOLDERS CMAKE_CXX_STANDARD_REQUIRED CXX_EXTENSIONS)
foreach(property IN LISTS consumer_properties)
  get_property(before_${property} GLOBAL PROPERTY ${property})
endforeach()

find_package(mqt-core CONFIG REQUIRED)

foreach(setting IN LISTS consumer_settings)
  get_property(
    after_cache
    CACHE ${setting}
    PROPERTY VALUE)
  get_property(
    after_cache_type
    CACHE ${setting}
    PROPERTY TYPE)
  if((defined_${setting} AND NOT DEFINED ${setting})
     OR (NOT defined_${setting} AND DEFINED ${setting})
     OR NOT "${${setting}}" STREQUAL "${before_${setting}}"
     OR NOT "${after_cache}" STREQUAL "${before_cache_${setting}}"
     OR NOT "${after_cache_type}" STREQUAL "${before_cache_type_${setting}}")
    message(FATAL_ERROR "find_package(mqt-core) changed ${setting}")
  endif()
endforeach()
get_directory_property(after_compile_options COMPILE_OPTIONS)
if(NOT before_compile_options STREQUAL after_compile_options)
  message(FATAL_ERROR "find_package(mqt-core) changed consumer compile options")
endif()
foreach(property IN LISTS consumer_properties)
  get_property(after GLOBAL PROPERTY ${property})
  if(NOT "${after}" STREQUAL "${before_${property}}")
    message(FATAL_ERROR "find_package(mqt-core) changed global property ${property}")
  endif()
endforeach()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/device.c" "int consumer_device(void) { return 0; }\n")
add_library(consumer-device MODULE "${CMAKE_CURRENT_BINARY_DIR}/device.c")
target_link_libraries(consumer-device PRIVATE qdmi::qdmi)
mqt_configure_qdmi_device(consumer-device ID test.consumer PREFIX CONSUMER)
add_custom_command(
  TARGET consumer-device
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E compare_files
    "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/consumer-device.qdmi.json"
    "$<TARGET_FILE_DIR:consumer-device>/consumer-device.qdmi.json")
