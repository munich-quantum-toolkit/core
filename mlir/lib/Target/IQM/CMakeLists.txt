# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(MLIRIQMTarget Export.cpp LINK_LIBS MQTCompilerTarget LLVMCore LLVMAnalysis)

target_link_libraries(MLIRIQMTarget PRIVATE LLVMSupport)
mqt_mlir_target_use_project_options(MLIRIQMTarget)
target_sources(
  MLIRIQMTarget
  PUBLIC
  FILE_SET HEADERS
    BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
    FILES ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Target/IQM/Export.h)
