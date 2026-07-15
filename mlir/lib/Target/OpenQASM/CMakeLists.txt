# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIROpenQASMTarget
  Frontend.cpp
  OpenQASM.cpp
  DEPENDS
  MLIROQ3OpsIncGen
  LINK_LIBS
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIROQ3Dialect
  MLIRQCDialect
  MLIRQCProgramBuilder
  MLIRSCFDialect
  MQT::CoreIR
  MQT::CoreQASM)

mqt_mlir_target_use_project_options(MLIROpenQASMTarget)

file(GLOB_RECURSE OPENQASM_HEADERS_SOURCE ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Target/OpenQASM/*.h)

target_sources(MLIROpenQASMTarget PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
                                         FILES ${OPENQASM_HEADERS_SOURCE})
