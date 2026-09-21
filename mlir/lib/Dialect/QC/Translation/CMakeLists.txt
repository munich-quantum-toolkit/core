# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIRQCTranslationSupport
  PARTIAL_SOURCES_INTENDED
  StandardGate.cpp
  MeasurementStores.cpp
  LINK_LIBS
  MLIRQCDialect
  MLIRCBitDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces)

mqt_mlir_target_use_project_options(MLIRQCTranslationSupport)

target_sources(
  MLIRQCTranslationSupport
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_SOURCE_INCLUDE_DIR}
         FILES
         ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Dialect/QC/Translation/StandardGate.h
         ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Dialect/QC/Translation/MeasurementStores.h)

add_mlir_library(
  MLIRQCOpenQASMTranslation
  PARTIAL_SOURCES_INTENDED
  OpenQASMToQCEmitter.cpp
  TranslateOpenQASMToQC.cpp
  TranslateQCToOpenQASM3.cpp
  LINK_LIBS
  MQTCompilerTarget
  MLIRAnalysis
  MLIRAffineToStandard
  MLIRArithDialect
  MLIRCBitDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIROpenQASMFrontend
  MLIRControlFlowDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRMQTDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRUBDialect
  MLIRQCDialect
  MLIRQCProgramBuilder
  MLIRTransforms
  MLIRQCTranslationSupport)

mqt_mlir_target_use_project_options(MLIRQCOpenQASMTranslation)

target_sources(
  MLIRQCOpenQASMTranslation
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_SOURCE_INCLUDE_DIR}
         FILES
         ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Dialect/QC/Translation/TranslateOpenQASMToQC.h
         ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Dialect/QC/Translation/TranslateQCToOpenQASM3.h)
