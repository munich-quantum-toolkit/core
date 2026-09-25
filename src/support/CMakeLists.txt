# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mqt_core_library(${MQT_CORE_TARGET_NAME}-support ALIAS_NAME Support)
target_link_libraries(${MQT_CORE_TARGET_NAME}-support PUBLIC LLVMSupport)
target_include_directories(
  ${MQT_CORE_TARGET_NAME}-support SYSTEM PUBLIC "$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>"
                                                "$<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>")
mqt_target_disable_exceptions(${MQT_CORE_TARGET_NAME}-support)
target_sources(${MQT_CORE_TARGET_NAME}-support PRIVATE Diagnostics.cpp DiagnosticFormatting.cpp)
mqt_source_enable_exceptions(DiagnosticFormatting.cpp)
target_sources(
  ${MQT_CORE_TARGET_NAME}-support
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_CORE_INCLUDE_BUILD_DIR}
         FILES
         ${MQT_CORE_INCLUDE_BUILD_DIR}/support/Diagnostics.hpp
         ${MQT_CORE_INCLUDE_BUILD_DIR}/support/DiagnosticFormatting.hpp)
include(GenerateExportHeader)
generate_export_header(${MQT_CORE_TARGET_NAME}-support BASE_NAME mqt_core_support)
target_sources(
  ${MQT_CORE_TARGET_NAME}-support
  PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/.. FILES
         ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_support_export.h)
if(NOT BUILD_MQT_CORE_SHARED_LIBS)
  target_compile_definitions(${MQT_CORE_TARGET_NAME}-support PUBLIC MQT_CORE_SUPPORT_STATIC_DEFINE)
endif()
set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-support
    PARENT_SCOPE)
