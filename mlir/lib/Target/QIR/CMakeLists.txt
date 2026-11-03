# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIRQIRTranslation
  Capabilities.cpp
  Export.cpp
  LINK_LIBS
  LLVMBitWriter
  MLIRIR
  MLIRLLVMDialect
  MLIRMQTDialect
  MLIRQIRUtils
  MLIRTargetLLVMIRExport)

mqt_mlir_target_use_project_options(MLIRQIRTranslation)

target_sources(
  MLIRQIRTranslation
  PUBLIC
  FILE_SET HEADERS
    BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
    FILES ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Target/QIR/Export.h
          ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Target/QIR/Capabilities.h)
