# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect_library(
  MLIRCUDAQuakeCompatDialect
  CCCompatOps.cpp
  CUDAQuakeCompatOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/mlir/Dialect/CUDAQuake
  DEPENDS
  MLIRQuakeCompatOpsIncGen
  MLIRCCCompatOpsIncGen
  LINK_LIBS
  PRIVATE
  MLIRIR)

mqt_mlir_target_use_project_options(MLIRCUDAQuakeCompatDialect)

file(GLOB_RECURSE IR_HEADERS_SOURCE
     "${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/CUDAQuake/IR/*.h")
file(GLOB_RECURSE IR_HEADERS_BUILD
     "${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Dialect/CUDAQuake/IR/*.inc")

target_sources(
  MLIRCUDAQuakeCompatDialect
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
