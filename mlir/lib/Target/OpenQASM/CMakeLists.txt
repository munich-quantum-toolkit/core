# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIROpenQASMFrontend
  Frontend.cpp
  GateCatalog.cpp
  OpenQASMLexer.cpp
  OpenQASMSemantics.cpp
  OpenQASMSyntax.cpp
  LINK_LIBS
  LLVMSupport)

mqt_mlir_target_use_project_options(MLIROpenQASMFrontend)

file(GLOB_RECURSE OPENQASM_HEADERS_SOURCE ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Target/OpenQASM/*.h)

target_sources(
  MLIROpenQASMFrontend PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                              ${OPENQASM_HEADERS_SOURCE})
