# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect_library(
  MLIRQuartzDialect
  QuartzDialect.cpp
  QuartzOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/mlir/Dialect/Quartz
  DEPENDS
  MLIRQuartzOpsIncGen
  MLIRQuartzInterfacesIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces)
