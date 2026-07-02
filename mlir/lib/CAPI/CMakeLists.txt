# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_public_c_api_library(
  MQTCoreCAPI
  Registration.cpp
  LINK_LIBS
  PUBLIC
  # Dialects
  MLIRQCDialect
  MLIRQCODialect
  MLIRQTensorDialect
  MLIRJeff
  # Conversions
  MLIRQCToQCO
  MLIRQCOToQC
  MLIRQCToQIRBase
  MLIRQCToQIRAdaptive
  MLIRJeffToQCO
  MLIRQCOToJeff
  # Transformations
  MLIRQCTransforms
  MLIRQCOTransforms
  MLIRQIRTransforms
  MLIRQTensorTransforms
  # OpenQASM 3 import
  MLIRQCTranslation
  MQT::CoreQASM
  MQT::CoreIR
  # Upstream
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect)

mqt_mlir_target_use_project_options(MQTCoreCAPI)
