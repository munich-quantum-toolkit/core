# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright ...

file(GLOB QIR_COMMON_SOURCES *.cpp)

add_mlir_library(
  MLIRQCToQIRCommon
  ${QIR_COMMON_SOURCES}
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRFuncToLLVM
  MLIRControlFlowToLLVM
  MLIRArithToLLVM
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRControlFlowDialect
  MLIRArithDialect
  MLIRQCDialect
  MLIRQIRUtils)

mqt_mlir_target_use_project_options(MLIRQCToQIRCommon)

file(GLOB_RECURSE QIR_COMMON_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Conversion/QCToQIR/QIRCommon/*.h)

target_sources(MLIRQCToQIRCommon PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
                                        FILES ${QIR_COMMON_HEADERS_SOURCE})
