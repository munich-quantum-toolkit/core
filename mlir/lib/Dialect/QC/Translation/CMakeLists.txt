# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MLIRQCTranslation
  TranslateQuantumComputationToQC.cpp
  LINK_LIBS
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRQCDialect
  MLIRQCProgramBuilder
  MQT::CoreIR
  DISABLE_INSTALL)

mqt_mlir_target_use_project_options(MLIRQCTranslation)

# collect header files
file(GLOB_RECURSE TRANSLATION_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QC/Translation/*.h)

# add public headers using file sets
target_sources(MLIRQCTranslation PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
                                        FILES ${TRANSLATION_HEADERS_SOURCE})
