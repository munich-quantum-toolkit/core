# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIRQCOProgramBuilder
  QCOProgramBuilder.cpp
  LINK_LIBS
  PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRQCODialect
  DISABLE_INSTALL)

mqt_mlir_target_use_project_options(MLIRQCOProgramBuilder)

# collect header files
file(GLOB_RECURSE BUILDER_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QCO/Builder/*.h)

# add public headers using file sets
target_sources(
  MLIRQCOProgramBuilder PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                               ${BUILDER_HEADERS_SOURCE})
