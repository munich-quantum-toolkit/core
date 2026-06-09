# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qir-jit)

if(NOT TARGET ${TARGET_NAME})
  # Add QIRJIT library
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME QIRJIT)

  # Add sources to target
  target_sources(${TARGET_NAME} PRIVATE Session.cpp)

  # Add headers using file sets
  target_sources(${TARGET_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                       FILES ${MQT_CORE_INCLUDE_BUILD_DIR}/qir/jit/Session.hpp)

  # Get the LLVM native target libraries
  llvm_map_components_to_libnames(
    llvm_native_libs
    codegen
    core
    executionengine
    irreader
    native
    orcdebugging
    orcjit
    orctargetprocess
    support
    targetparser)

  # Add link libraries
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQIRRuntime ${llvm_native_libs})
endif()
