# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIRMQTUtils
  Angles.cpp
  ConstantFolding.cpp
  DenseUnitary.cpp
  GatePowering.cpp
  Modifiers.cpp
  Parameters.cpp
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/MQT/Utils
  LINK_LIBS
  PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRSideEffectInterfaces)

mqt_mlir_target_use_project_options(MLIRMQTUtils)

file(GLOB UTILS_HEADERS_SOURCE "${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/MQT/Utils/*.h")
target_sources(MLIRMQTUtils PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                                   ${UTILS_HEADERS_SOURCE})
