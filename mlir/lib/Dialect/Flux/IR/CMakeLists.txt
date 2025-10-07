# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect_library(
  MLIRFlux
  FluxOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/mlir/Dialect/Flux
  DEPENDS
  MLIRFluxOpsIncGen
  MLIRFluxInterfacesIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces)

# collect header files
file(GLOB_RECURSE IR_HEADERS_SOURCE "${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/Flux/IR/*.h")
file(GLOB_RECURSE IR_HEADERS_BUILD "${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Dialect/Flux/IR/*.inc")

# add public headers using file sets
target_sources(
  MLIRFlux
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
