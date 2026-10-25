# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIROpenQASMFrontend
  PARTIAL_SOURCES_INTENDED
  Frontend.cpp
  GateCatalog.cpp
  OpenQASMLexer.cpp
  OpenQASMSemantics.cpp
  OpenQASMSyntax.cpp
  LINK_LIBS
  LLVMSupport
  MLIRPresburger
  MLIRQCTranslationSupport)

mqt_mlir_target_use_project_options(MLIROpenQASMFrontend)

set(OPENQASM_HEADERS_SOURCE ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Target/OpenQASM/Frontend.h
    ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Target/OpenQASM/GateCatalog.h)
file(GLOB_RECURSE OPENQASM_DETAIL_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Target/OpenQASM/Detail/*.h)

target_sources(
  MLIROpenQASMFrontend
  PUBLIC
  FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES ${OPENQASM_HEADERS_SOURCE})

target_sources(MLIROpenQASMFrontend PRIVATE ${OPENQASM_DETAIL_HEADERS_SOURCE})

add_mlir_library(
  MLIRQCOpenQASMTranslation
  PARTIAL_SOURCES_INTENDED
  OpenQASMToQCEmitter.cpp
  TranslateOpenQASMToQC.cpp
  TranslateQCToOpenQASM3.cpp
  LINK_LIBS
  MQTCompilerTarget
  MLIRAnalysis
  MLIRFuncInlinerExtension
  MLIRLLVMIRTransforms
  MLIRTransforms
  MLIRAffineToStandard
  MLIRArithDialect
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRCBitDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIROpenQASMFrontend
  MLIRControlFlowDialect
  MLIRMathDialect
  MLIRBufferizationDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRMQTDialect
  MLIRMQTUtils
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRUBDialect
  MLIRQCDialect
  MLIRQCProgramBuilder
  MLIRValueBoundsOpInterface
  MLIRQCTranslationSupport)

mqt_mlir_target_use_project_options(MLIRQCOpenQASMTranslation)

target_sources(
  MLIRQCOpenQASMTranslation
  PUBLIC
  FILE_SET HEADERS
    BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
    FILES ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Target/OpenQASM/TranslateOpenQASMToQC.h
          ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Target/OpenQASM/TranslateQCToOpenQASM3.h)
