# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIRMQTTransforms
  NormalizeGlobalPhases.cpp
  UnrollModifiers.cpp
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/Utils/Transforms
  LINK_LIBS
  PRIVATE
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRQCDialect
  MLIRQCODialect
  MLIRSideEffectInterfaces
  DEPENDS
  MLIRMQTTransformsIncGen)

mqt_mlir_target_use_project_options(MLIRMQTTransforms)

file(GLOB_RECURSE PASSES_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/Utils/Transforms/*.h)
file(GLOB_RECURSE PASSES_HEADERS_BUILD
     ${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Dialect/Utils/Transforms/*.inc)

target_sources(
  MLIRMQTTransforms
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_SOURCE_INCLUDE_DIR}
         FILES
         ${PASSES_HEADERS_SOURCE}
         FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_BUILD_INCLUDE_DIR}
         FILES
         ${PASSES_HEADERS_BUILD})
