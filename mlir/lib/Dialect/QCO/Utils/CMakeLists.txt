# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(MLIRQCOMatrix PARTIAL_SOURCES_INTENDED Matrix.cpp LINK_LIBS PUBLIC MLIRSupport)

mqt_mlir_target_use_project_options(MLIRQCOMatrix)

target_sources(MLIRQCOMatrix PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                                    ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QCO/Utils/Matrix.h)

file(GLOB_RECURSE UTILS_CPP "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(FILTER UTILS_CPP EXCLUDE REGEX "Matrix\\.cpp$")

add_mlir_dialect_library(
  MLIRQCOUtils
  PARTIAL_SOURCES_INTENDED
  ${UTILS_CPP}
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/mlir/Dialect/QCO
  DEPENDS
  MLIRQCOOpsIncGen
  MLIRQCOInterfacesIncGen
  LINK_LIBS
  PUBLIC
  MLIRQCODialect
  MLIRSCFDialect)

mqt_mlir_target_use_project_options(MLIRQCOUtils)

# collect header files
file(GLOB_RECURSE UTILS_HEADERS_SOURCE "${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QCO/Utils/*.h")
list(FILTER UTILS_HEADERS_SOURCE EXCLUDE REGEX "Matrix\\.h$")
file(GLOB_RECURSE UTILS_HEADERS_BUILD "${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Dialect/QCO/Utils/*.inc")

# add public headers using file sets
target_sources(
  MLIRQCOUtils
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_SOURCE_INCLUDE_DIR}
         FILES
         ${UTILS_HEADERS_SOURCE}
         FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_BUILD_INCLUDE_DIR}
         FILES
         ${UTILS_HEADERS_BUILD})
