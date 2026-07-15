# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect_library(
  MLIROQ3Transforms
  LowerOQ3ToQC.cpp
  DEPENDS
  MLIROQ3PassIncGen
  LINK_LIBS
  PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIROQ3Dialect
  MLIRQCDialect
  MLIRSCFDialect
  MLIRTransformUtils)

mqt_mlir_target_use_project_options(MLIROQ3Transforms)
