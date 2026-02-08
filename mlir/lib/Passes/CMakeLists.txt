# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB_RECURSE PASSES_SOURCES *.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(
  QcoPasses
  ${PASSES_SOURCES}
  LINK_LIBS
  PUBLIC
  MQT::CoreIR Eigen3::Eigen
  PRIVATE
  ${dialect_libs}
  PRIVATE
  DEPENDS
  QcoPassesIncGen)

# collect header files
file(GLOB_RECURSE PASSES_HEADERS_SOURCE ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Passes/*.h)
file(GLOB_RECURSE PASSES_HEADERS_BUILD ${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Passes/*.inc)

# add public headers using file sets
target_sources(
  QcoPasses
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

# fix alignment issues for stack-allocated fixed-size matrices
target_compile_definitions(QcoPasses PUBLIC EIGEN_MAX_STATIC_ALIGN_BYTES=16)
