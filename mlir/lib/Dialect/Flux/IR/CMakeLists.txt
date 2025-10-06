# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect_library(
  MLIRFluxDialect
  FluxDialect.cpp
  FluxOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/mlir/Dialect/Flux
  DEPENDS
  MLIRFluxOpsIncGen
  MLIRFluxInterfacesIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces)
