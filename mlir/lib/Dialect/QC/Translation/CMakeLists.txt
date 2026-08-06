# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIRQCOpenQASMTranslation
  PARTIAL_SOURCES_INTENDED
  OpenQASMToQCEmitter.cpp
  TranslateQASM3ToQC.cpp
  TranslateQCToOpenQASM3.cpp
  LINK_LIBS
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIROpenQASMFrontend
  MLIRControlFlowDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRUBDialect
  MLIRQCDialect
  MLIRQCProgramBuilder)

mqt_mlir_target_use_project_options(MLIRQCOpenQASMTranslation)

target_sources(
  MLIRQCOpenQASMTranslation
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_SOURCE_INCLUDE_DIR}
         FILES
         ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h
         ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QC/Translation/TranslateQCToOpenQASM3.h)

add_mlir_library(
  MLIRQCTranslation
  PARTIAL_SOURCES_INTENDED
  TranslateQuantumComputationToQC.cpp
  LINK_LIBS
  PUBLIC
  MLIRQCOpenQASMTranslation
  MLIRQCDialect
  MLIRQCProgramBuilder
  MQT::CoreIR)

mqt_mlir_target_use_project_options(MLIRQCTranslation)

target_sources(
  MLIRQCTranslation
  PUBLIC
    FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
    ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h)
