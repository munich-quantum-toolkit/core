# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB_RECURSE PASSES_SOURCES *.cpp)

add_mlir_library(
  MLIRQTensorTransforms
  ${PASSES_SOURCES}
  LINK_LIBS
  PRIVATE
  MLIRQTensorDialect
  DEPENDS
  MLIRQTensorTransformsIncGen)

# collect header files
file(GLOB_RECURSE PASSES_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QTensor/Transforms/*.h)
file(GLOB_RECURSE PASSES_HEADERS_BUILD
     ${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Dialect/QTensor/Transforms/*.inc)

# add public headers using file sets
target_sources(
  MLIRQTensorTransforms
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
