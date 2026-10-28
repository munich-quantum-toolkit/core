# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIRQCTranslationSupport
  StandardGate.cpp
  MeasurementStores.cpp
  LINK_LIBS
  MLIRQCDialect
  MLIRCBitDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces)

mqt_mlir_target_use_project_options(MLIRQCTranslationSupport)

target_sources(
  MLIRQCTranslationSupport
  PUBLIC
  FILE_SET HEADERS
    BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
    FILES ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Dialect/QC/Translation/StandardGate.h
          ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Dialect/QC/Translation/MeasurementStores.h)
