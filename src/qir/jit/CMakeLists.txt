# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qir-jit)

if(NOT TARGET ${TARGET_NAME})
  # Add the internal QIR JIT library.
  add_library(${TARGET_NAME} STATIC)

  # Add sources to target
  target_sources(
    ${TARGET_NAME}
    PRIVATE Session.cpp IRRewriter.cpp ${PROJECT_SOURCE_DIR}/src/qir/include/qir/Definitions.hpp
            ${PROJECT_SOURCE_DIR}/src/qir/include/qir/jit/Session.hpp
            ${PROJECT_SOURCE_DIR}/src/qir/include/qir/jit/IRRewriter.hpp)
  target_include_directories(${TARGET_NAME}
                             PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/qir/include>)
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_20)
  set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  mqt_llvm_target_disable_rtti(${TARGET_NAME})

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
    targetparser
    transformutils)

  # Add link libraries
  target_link_libraries(${TARGET_NAME} PRIVATE mqt-core-qir-runtime MQT::ProjectWarnings
                                               MQT::ProjectOptions ${llvm_native_libs})
endif()
