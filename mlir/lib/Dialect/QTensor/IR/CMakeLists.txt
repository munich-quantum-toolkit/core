# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB_RECURSE OPERATIONS "${CMAKE_CURRENT_SOURCE_DIR}/Operations/*.cpp")

add_mlir_dialect_library(
  MLIRQTensorDialect
  QTensorOps.cpp
  ${OPERATIONS}
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QTensor
  DEPENDS
  MLIRQTensorOpsIncGen
  LINK_LIBS
  PRIVATE
  MLIRIR
  MLIRDialectUtils
  MLIRArithDialect
  MLIRDialectUtils
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  PUBLIC
  MLIRQCODialect)

mqt_mlir_target_use_project_options(MLIRQTensorDialect)

# collect header files
file(GLOB_RECURSE IR_HEADERS_SOURCE "${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QTensor/IR/*.h")
file(GLOB_RECURSE IR_HEADERS_BUILD "${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Dialect/QTensor/IR/*.inc")

# add public headers using file sets
target_sources(
  MLIRQTensorDialect
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
