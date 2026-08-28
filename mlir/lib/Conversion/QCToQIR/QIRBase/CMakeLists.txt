# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
  MLIRQCToQIRBase
  ${CONVERSION_SOURCES}
  DEPENDS
  QCToQIRBaseIncGen
  LINK_LIBS
  MLIRCBitDialect
  MLIRQIRUtils
  MLIRQCToQIRCommon
  MLIRMQTDialect
  MLIRMQTTransforms
  MLIRLLVMDialect
  MLIRQCDialect
  MLIRQCODialect
  MLIRArithDialect
  MLIRTransforms
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRReconcileUnrealizedCasts)

mqt_mlir_target_use_project_options(MLIRQCToQIRBase)
