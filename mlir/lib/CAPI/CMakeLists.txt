# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_public_c_api_library(
  MQTMLIRCoreDialectsCAPI
  Dialects.cpp
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/CAPI
  ENABLE_AGGREGATION
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRSupport
  MLIRQCDialect
  MLIRQCODialect
  MLIRQTensorDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRQCTransforms
  MLIRQCOTransforms
  MLIRQCToQCO)

mqt_mlir_target_use_project_options(MQTMLIRCoreDialectsCAPI)
