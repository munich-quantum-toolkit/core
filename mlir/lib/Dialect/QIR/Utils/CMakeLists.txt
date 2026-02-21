# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB QIR_UTILS_SOURCES *.cpp)

add_mlir_library(
  MLIRQIRUtils
  ${QIR_UTILS_SOURCES}
  LINK_LIBS
  PUBLIC
  MLIRLLVMDialect
  MLIRIR
  DISABLE_INSTALL)

mqt_mlir_target_use_project_options(MLIRQIRUtils)

file(GLOB_RECURSE QIR_UTILS_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QIR/Utils/*.h)

target_sources(MLIRQIRUtils PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                                   ${QIR_UTILS_HEADERS_SOURCE})
