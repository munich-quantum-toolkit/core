# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_library(
  QIRToMQTRef
  ${CONVERSION_SOURCES}
  DEPENDS
  QIRToMQTRefIncGen
  LINK_LIBS
  MLIRMQTRef
  MLIRMemRefDialect
  MLIRTransforms
  MLIRLLVMDialect
  DISABLE_INSTALL)
