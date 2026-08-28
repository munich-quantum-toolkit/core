# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_conversion_library(
  MLIRCBitToMemRef
  CBitToMemRef.cpp
  DEPENDS
  CBitToMemRefIncGen
  LINK_LIBS
  MLIRArithDialect
  MLIRCBitDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms)

mqt_mlir_target_use_project_options(MLIRCBitToMemRef)
