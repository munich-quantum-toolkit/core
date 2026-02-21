# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB_RECURSE MODIFIERS "${CMAKE_CURRENT_SOURCE_DIR}/Modifiers/*.cpp")
file(GLOB_RECURSE OPERATIONS "${CMAKE_CURRENT_SOURCE_DIR}/Operations/*.cpp")
file(GLOB_RECURSE QUBIT_MANAGEMENT "${CMAKE_CURRENT_SOURCE_DIR}/QubitManagement/*.cpp")

add_mlir_dialect_library(
  MLIRQCODialect
  QCOOps.cpp
  ${MODIFIERS}
  ${OPERATIONS}
  ${QUBIT_MANAGEMENT}
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/mlir/Dialect/QCO
  DEPENDS
  MLIRQCOOpsIncGen
  MLIRQCOInterfacesIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  Eigen3::Eigen
  DISABLE_INSTALL)

mqt_mlir_target_use_project_options(MLIRQCODialect)

# collect header files
file(GLOB_RECURSE IR_HEADERS_SOURCE "${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QCO/IR/*.h")
file(GLOB_RECURSE IR_HEADERS_BUILD "${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Dialect/QCO/IR/*.inc")

# add public headers using file sets
target_sources(
  MLIRQCODialect
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

# fix alignment issues for stack-allocated fixed-size matrices
target_compile_definitions(MLIRQCODialect PUBLIC EIGEN_MAX_STATIC_ALIGN_BYTES=16)
