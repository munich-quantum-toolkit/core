# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
  MLIRQCToQIRAdaptive
  ${CONVERSION_SOURCES}
  DEPENDS
  QCToQIRAdaptiveIncGen
  LINK_LIBS
  MLIRQCToQIRCommon
  MLIRQIRUtils
  MLIRLLVMDialect
  MLIRQCDialect
  MLIRArithDialect
  MLIRTransforms
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRSCFToControlFlow
  MLIRReconcileUnrealizedCasts)

mqt_mlir_target_use_project_options(MLIRQCToQIRAdaptive)
