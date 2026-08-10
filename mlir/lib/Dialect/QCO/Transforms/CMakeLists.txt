# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB_RECURSE PASSES_SOURCES *.cpp)
set_source_files_properties(Decomposition/BasisDecomposer.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION
                                                                         ON)

add_mlir_library(
  MLIRQCOTransforms
  ${PASSES_SOURCES}
  LINK_LIBS
  PUBLIC
  MQTCompilerTarget
  PRIVATE
  MLIRQCODialect
  MLIRQCOUtils
  MLIRQTensorUtils
  MLIRArithDialect
  MLIRMathDialect
  MLIRMQTTransforms
  MLIRSCFUtils
  DEPENDS
  MLIRQCOTransformsIncGen)

mqt_mlir_target_use_project_options(MLIRQCOTransforms)

# collect header files
file(GLOB_RECURSE PASSES_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QCO/Transforms/*.h)
file(GLOB_RECURSE PASSES_HEADERS_BUILD
     ${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Dialect/QCO/Transforms/*.inc)

# add public headers using file sets
target_sources(
  MLIRQCOTransforms
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
