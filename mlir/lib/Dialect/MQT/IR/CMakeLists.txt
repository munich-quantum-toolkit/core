# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect_library(
  MLIRMQTDialect
  MQTDialect.cpp
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/MQT
  DEPENDS
  MLIRMQTDialectIncGen
  LINK_LIBS
  PRIVATE
  MLIRCBitDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRQCDialect
  MLIRQCODialect
  MLIRSCFDialect
  MLIRQTensorDialect)

mqt_mlir_target_use_project_options(MLIRMQTDialect)

file(GLOB_RECURSE IR_HEADERS_SOURCE "${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/MQT/IR/*.h")
file(GLOB_RECURSE IR_HEADERS_BUILD "${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Dialect/MQT/IR/*.inc")

target_sources(
  MLIRMQTDialect
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
