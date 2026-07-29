# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.24...4.2)

project(mqt-core-qdmi-consumer LANGUAGES C CXX)

if(NOT MQT_CORE_SOURCE_DIR)
  message(FATAL_ERROR "MQT_CORE_SOURCE_DIR must point to the MQT Core source tree")
endif()

if(EXISTS "${PARENT_SPDLOG_SOURCE_DIR}/CMakeLists.txt")
  set(SPDLOG_BUILD_PIC ON)
  set(SPDLOG_INSTALL OFF)
  add_subdirectory("${PARENT_SPDLOG_SOURCE_DIR}" spdlog EXCLUDE_FROM_ALL)
elseif(PARENT_SPDLOG_DIR)
  set(spdlog_DIR "${PARENT_SPDLOG_DIR}")
  find_package(spdlog CONFIG REQUIRED)
else()
  message(FATAL_ERROR "A parent spdlog source or package directory is required")
endif()

set(MQT_CORE_INSTALL
    OFF
    CACHE BOOL "")
set(BUILD_MQT_CORE_BINDINGS
    OFF
    CACHE BOOL "")
set(BUILD_MQT_CORE_TESTS
    OFF
    CACHE BOOL "")
set(BUILD_MQT_CORE_BENCHMARKS
    OFF
    CACHE BOOL "")
set(BUILD_MQT_CORE_DOCUMENTATION
    OFF
    CACHE BOOL "")
add_subdirectory("${MQT_CORE_SOURCE_DIR}" mqt-core EXCLUDE_FROM_ALL)

get_target_property(fomac_binary_program_api MQT::CoreFoMaC MQT_CORE_FOMAC_BINARY_PROGRAM_API)
if(NOT fomac_binary_program_api)
  message(FATAL_ERROR "FoMaC does not advertise exact-byte program submission")
endif()

add_executable(mqt-core-qdmi-consumer main.cpp)
target_link_libraries(mqt-core-qdmi-consumer PRIVATE MQT::CoreFoMaC MQT::CoreQDMI_DDSIM_Device)
