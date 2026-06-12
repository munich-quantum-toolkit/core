# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_public_c_api_library(
  MQTCoreCAPI
  Dialects.cpp
  LINK_LIBS
  PUBLIC
  MLIRQCDialect
  MLIRQCODialect
  MLIRQTensorDialect
  MLIRQCToQCO
  MLIRQCTranslation
  MQT::CoreQASM
  MQT::CoreIR
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect)

mqt_mlir_target_use_project_options(MQTCoreCAPI)
