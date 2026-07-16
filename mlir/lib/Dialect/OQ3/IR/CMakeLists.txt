# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect_library(
  MLIROQ3Dialect
  GateCatalog.cpp
  OQ3Ops.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/mlir/Dialect/OQ3
  DEPENDS
  MLIROQ3EnumsIncGen
  MLIROQ3OpsIncGen
  LINK_LIBS
  PRIVATE
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRQCDialect
  MLIRSideEffectInterfaces)

mqt_mlir_target_use_project_options(MLIROQ3Dialect)

file(GLOB_RECURSE IR_HEADERS_SOURCE "${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/OQ3/IR/*.h")
file(GLOB_RECURSE IR_HEADERS_BUILD "${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Dialect/OQ3/IR/*.inc")

target_sources(
  MLIROQ3Dialect
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_SOURCE_INCLUDE_DIR}
         FILES
         ${IR_HEADERS_SOURCE}
         FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_BUILD_INCLUDE_DIR}
         FILES
         ${IR_HEADERS_BUILD})
