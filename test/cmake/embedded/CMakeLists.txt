# Copyright (c) 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.24...4.2)

project(mqt-core-embedded-consumer LANGUAGES C CXX)

if(NOT MQT_CORE_SOURCE_DIR)
  message(FATAL_ERROR "MQT_CORE_SOURCE_DIR must point to an MQT Core source tree")
endif()

# Mirror a larger LLVM project that already owns MLIR's aggregate documentation target and a global
# test target before embedding MQT Core.
find_package(MLIR REQUIRED CONFIG)
add_custom_target(check)
set(BUILD_TESTING ON)
set(BUILD_MQT_CORE_QDMI_DEVICES
    ${EMBEDDED_BUILD_MQT_CORE_QDMI_DEVICES}
    CACHE BOOL "" FORCE)

add_subdirectory(${MQT_CORE_SOURCE_DIR} mqt-core EXCLUDE_FROM_ALL)

if(NOT TARGET MQT::CoreFoMaC)
  message(FATAL_ERROR "The embedded MQT Core project did not provide MQT::CoreFoMaC")
endif()

function(assert_qdmi_device target expected_id expected_prefix)
  if(EMBEDDED_BUILD_MQT_CORE_QDMI_DEVICES)
    if(NOT TARGET ${target})
      message(FATAL_ERROR "Embedded MQT Core did not provide bundled QDMI device ${target}")
    endif()
    get_target_property(device_id ${target} QDMI_DEVICE_ID)
    get_target_property(device_prefix ${target} QDMI_DEVICE_PREFIX)
    if(NOT device_id STREQUAL expected_id OR NOT device_prefix STREQUAL expected_prefix)
      message(FATAL_ERROR "Bundled QDMI device ${target} has invalid registration metadata")
    endif()
  elseif(TARGET ${target})
    message(FATAL_ERROR "Embedded MQT Core unexpectedly built bundled QDMI device ${target}")
  endif()
endfunction()

assert_qdmi_device(MQT::CoreQDMI_DDSIM_Device mqt.ddsim.default MQT_DDSIM)
assert_qdmi_device(MQT::CoreQDMIScDevice mqt.sc.default MQT_SC)
assert_qdmi_device(MQT::CoreQDMINaDevice mqt.na.default MQT_NA)
