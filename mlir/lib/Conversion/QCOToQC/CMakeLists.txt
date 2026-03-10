# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_library(
  QCOToQC
  ${CONVERSION_SOURCES}
  DEPENDS
  QCOToQCIncGen
  LINK_LIBS
  MLIRQCDialect
  MLIRQCODialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRFuncTransforms
  DISABLE_INSTALL)

mqt_mlir_target_use_project_options(QCOToQC)
