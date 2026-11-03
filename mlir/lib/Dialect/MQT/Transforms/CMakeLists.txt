# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIRMQTTransforms
  SimplifyQuantum.cpp
  DropGlobalPhases.cpp
  NormalizeGlobalPhases.cpp
  LegalizePayloadControlFlow.cpp
  PromoteClassicalMemory.cpp
  SimplifyClassicalControl.cpp
  UnrollModifiers.cpp
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Dialect/MQT/Transforms
  LINK_LIBS
  PRIVATE
  MQTCompilerTarget
  MLIRArithDialect
  MLIRAsmParser
  MLIRCBitDialect
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRMathDialect
  MLIRMemorySlotInterfaces
  MLIRLLVMIRTransforms
  MLIRMQTUtils
  MLIRPass
  MLIRQCDialect
  MLIRQCODialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRTransformUtils
  DEPENDS
  MLIRMQTTransformsIncGen)

mqt_mlir_target_use_project_options(MLIRMQTTransforms)

file(GLOB_RECURSE PASSES_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Dialect/MQT/Transforms/*.h)
file(GLOB_RECURSE PASSES_HEADERS_BUILD
     ${MQT_MLIR_BUILD_INCLUDE_DIR}/mqt/Dialect/MQT/Transforms/*.inc)

target_sources(
  MLIRMQTTransforms
  PUBLIC
  FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES ${PASSES_HEADERS_SOURCE}
  FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_BUILD_INCLUDE_DIR} FILES ${PASSES_HEADERS_BUILD})
