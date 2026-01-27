# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_library(
  JeffToQC
  ${CONVERSION_SOURCES}
  DEPENDS
  JeffToQCIncGen
  LINK_LIBS
  MLIRJeff
  MLIRQCDialect
  MLIRTransforms
  DISABLE_INSTALL)
