# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Build the immutable compiler target model
add_mlir_library(
  MQTCompilerTarget
  PARTIAL_SOURCES_INTENDED
  Target.cpp
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Compiler
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRQCODialect)

mqt_mlir_target_use_project_options(MQTCompilerTarget)

target_sources(MQTCompilerTarget PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
                                        FILES ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Compiler/Target.h)

# Build the optional QDMI-to-compiler-target adapter
add_mlir_library(
  MQTCompilerQDMIAdapter
  PARTIAL_SOURCES_INTENDED
  QDMIAdapter.cpp
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Compiler
  LINK_LIBS
  PUBLIC
  MQTCompilerTarget
  MQT::CoreQDMI)

mqt_mlir_target_use_project_options(MQTCompilerQDMIAdapter)

target_sources(
  MQTCompilerQDMIAdapter PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                                ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Compiler/QDMIAdapter.h)

# Build the compiler pipeline library
add_mlir_library(
  MQTCompilerPipeline
  PARTIAL_SOURCES_INTENDED
  Programs.cpp
  TargetCompilation.cpp
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Compiler
  LINK_LIBS
  PUBLIC
  LLVMBitWriter
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRMQTTransforms
  MLIRQCToQCO
  MLIRQCOToQC
  MLIRQCOToJeff
  MLIRQCToQIRBase
  MLIRQCToQIRAdaptive
  MLIRQIRUtils
  MLIRQCOTransforms
  MLIRQCTranslation
  MLIRParser
  MLIRJeffTranslation
  MLIRJeffToQCO
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MQTCompilerTarget
  MQT::MLIRSupport)

mqt_mlir_target_use_project_options(MQTCompilerPipeline)

# collect header files
file(GLOB_RECURSE COMPILER_HEADERS_SOURCE "${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Compiler/*.h")
list(FILTER COMPILER_HEADERS_SOURCE EXCLUDE REGEX "/(QDMIAdapter|Target)\\.h$")

target_sources(MQTCompilerPipeline PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
                                          FILES ${COMPILER_HEADERS_SOURCE})
