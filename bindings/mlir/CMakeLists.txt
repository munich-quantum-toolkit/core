# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mqt_python_binding_nanobind(
  CORE
  ${MQT_CORE_TARGET_NAME}-mlir-bindings
  register_mlir.cpp
  MODULE_NAME
  mlir
  INSTALL_DIR
  .
  LINK_LIBS
  MQT::CoreIR
  MQT::CoreQASM
  MQTMLIRCoreDialectsCAPI
  MLIRQCTranslation
  MLIRSupportMQT
  MQTCompilerPipeline
  MLIRParser)
