# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIRQIRProgramBuilder
  QIRProgramBuilder.cpp
  LINK_LIBS
  PUBLIC
  MLIRLLVMDialect
  MLIRQIRUtils
  MLIRIR
  MLIRSupport)

# collect header files
file(GLOB_RECURSE QIR_BUILDER_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QIR/Builder/*.h)

# add public headers using file sets
target_sources(
  MLIRQIRProgramBuilder PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                               ${QIR_BUILDER_HEADERS_SOURCE})
