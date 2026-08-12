# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIRMQTAngleUtils
  AngleConversion.cpp
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/Utils
  LINK_LIBS
  PUBLIC
  MLIRArithDialect
  MLIRIR)

mqt_mlir_target_use_project_options(MLIRMQTAngleUtils)

target_sources(
  MLIRMQTAngleUtils PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                           ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/Utils/AngleConversion.h)

add_subdirectory(Transforms)
