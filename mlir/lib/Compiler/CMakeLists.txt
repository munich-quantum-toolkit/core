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
  MLIRMQTDialect
  MLIRQCODialect)

mqt_mlir_target_use_project_options(MQTCompilerTarget)

target_sources(MQTCompilerTarget PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
                                        FILES ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Compiler/Target.h)

# Build the optional QDMI-to-compiler-target adapter
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)
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
unset(LLVM_REQUIRES_EH)
unset(LLVM_REQUIRES_RTTI)

target_sources(
  MQTCompilerQDMIAdapter PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                                ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Compiler/QDMIAdapter.h)

# Build the compiler program model and OpenQASM-to-QCO path
add_mlir_library(
  MQTCompilerPrograms
  PARTIAL_SOURCES_INTENDED
  Programs.cpp
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Compiler
  LINK_LIBS
  PUBLIC
  LLVMSupport
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRCBitDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRJeff
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMQTDialect
  MLIRParser
  MLIRPass
  MLIRQCDialect
  MLIRQCOpenQASMTranslation
  MLIRQCToQCO
  MLIRQCODialect
  MLIRQTensorDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect)

mqt_mlir_target_use_project_options(MQTCompilerPrograms)

target_sources(
  MQTCompilerPrograms PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                             ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Compiler/Programs.h)

# Build the compiler pipeline library
add_mlir_library(
  MQTCompilerPipeline
  PARTIAL_SOURCES_INTENDED
  Pipeline.cpp
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
  MLIRQCOToQC
  MLIRQCOToJeff
  MLIRQCToQIRBase
  MLIRQCToQIRAdaptive
  MLIRQIRUtils
  MLIRQCOTransforms
  MLIRQCOpenQASMTranslation
  MLIRJeffTranslation
  MLIRJeffToQCO
  MLIRTargetLLVMIRExport
  MQTCompilerPrograms
  MQTCompilerTarget
  MQT::MLIRSupport)

mqt_mlir_target_use_project_options(MQTCompilerPipeline)

target_sources(
  MQTCompilerPipeline PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                             ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Compiler/TargetCompilation.h)
