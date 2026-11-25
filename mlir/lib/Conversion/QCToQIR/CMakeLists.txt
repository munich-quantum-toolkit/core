# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
  MLIRQCToQIR
  ${CONVERSION_SOURCES}
  DEPENDS
  QCToQIRIncGen
  LINK_LIBS
  PUBLIC
  MQTCompilerTarget
  MLIRMQTTransforms
  MLIRQIRTransforms
  MLIRQIRTranslation
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRArithToLLVM
  MLIRArithTransforms
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRBufferizationTransforms
  MLIRMemRefToLLVM
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRMemRefUtils
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRTensorTransforms
  MLIRMQTDialect
  MLIRFuncDialect
  MLIRControlFlowDialect
  MLIRArithDialect
  MLIRCBitDialect
  MLIRQCDialect
  MLIRQIRUtils)

mqt_mlir_target_use_project_options(MLIRQCToQIR)

target_sources(
  MLIRQCToQIR
  PUBLIC
  FILE_SET HEADERS
    BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
    FILES ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Conversion/QCToQIR/QIRCommon.h)
