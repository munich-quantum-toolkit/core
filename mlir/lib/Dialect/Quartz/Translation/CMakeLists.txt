# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_compile_options("-fexceptions")

add_mlir_library(
  MLIRQuartzTranslation
  TranslateQuantumComputationToQuartz.cpp
  LINK_LIBS
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRQuartzDialect
  MQT::CoreIR)

# collect header files
file(GLOB_RECURSE TRANSLATION_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/Quartz/Translation/*.h)

# add public headers using file sets
target_sources(
  MLIRQuartzTranslation PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                               ${TRANSLATION_HEADERS_SOURCE})
