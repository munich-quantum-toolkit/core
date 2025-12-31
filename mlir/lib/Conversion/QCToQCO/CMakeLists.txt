# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_library(
  QCToQCO
  ${CONVERSION_SOURCES}
  DEPENDS
  QCToQCOIncGen
  LINK_LIBS
  PUBLIC
  MLIRQCDialect
  MLIRQCODialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRFuncTransforms)
