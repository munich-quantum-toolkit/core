# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_compile_options(-fexceptions)

add_mlir_dialect_library(
  MLIRMQTDynImportQuantumComputation
  ImportQuantumComputation.cpp
  LINK_LIBS
  MQT::CoreIR
  DEPENDS
  MLIRIR
  MLIRSupport)

target_include_directories(MLIRMQTDynImportQuantumComputation PRIVATE ${MLIR_MAIN_INCLUDE_DIR})
