# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_library(
  MQTRefToMQTOpt
  ${CONVERSION_SOURCES}
  DEPENDS
  MQTRefToMQTOptIncGen
  LINK_LIBS
  MLIRMQTOpt
  MLIRMQTRef
  MLIRMemRefDialect
  MLIRTransforms
  MLIRFuncDialect
  MLIRFuncTransforms
  DISABLE_INSTALL)
