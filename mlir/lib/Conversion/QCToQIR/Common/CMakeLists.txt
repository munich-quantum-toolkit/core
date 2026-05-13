# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright ...

file(GLOB COMMON_QIR_SOURCES *.cpp)

add_mlir_library(
  MLIRQCToQIRCommon
  ${COMMON_QIR_SOURCES}
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

file(GLOB_RECURSE COMMON_QIR_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Conversion/QCToQIR/Common/*.h)

target_sources(MLIRQCToQIRCommon PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
                                        FILES ${COMMON_QIR_HEADERS_SOURCE})
