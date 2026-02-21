# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB_RECURSE UTILS_CPP "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_mlir_dialect_library(
  MLIRQCOUtils
  ${UTILS_CPP}
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/mlir/Dialect/QCO
  DEPENDS
  MLIRQCOOpsIncGen
  MLIRQCOInterfacesIncGen
  LINK_LIBS
  PUBLIC
  Eigen3::Eigen
  DISABLE_INSTALL)

mqt_mlir_target_use_project_options(MLIRQCOUtils)

# collect header files
file(GLOB_RECURSE UTILS_HEADERS_SOURCE "${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QCO/Utils/*.h")
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
