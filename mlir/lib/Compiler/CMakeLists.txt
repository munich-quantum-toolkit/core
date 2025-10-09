# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Build the compiler pipeline library
add_mlir_library(
  MQTCompilerPipeline
  CompilerPipeline.cpp
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Compiler
  LINK_LIBS
  PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  QuartzToFlux
  FluxToQuartz
  QuartzToQIR
  MQT::ProjectOptions)

# collect header files
file(GLOB_RECURSE COMPILER_HEADERS_SOURCE "${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Compiler/*.h")

target_sources(MQTCompilerPipeline PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
                                          FILES ${COMPILER_HEADERS_SOURCE})
