# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_library(
  QCToQIR
  ${CONVERSION_SOURCES}
  DEPENDS
  QCToQIRIncGen
  LINK_LIBS
  PUBLIC
  MLIRQIRUtils
  MLIRLLVMDialect
  MLIRQCDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts)
