# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB QIR_UTILS_SOURCES *.cpp)

add_mlir_library(MLIRQIRUtils ${QIR_UTILS_SOURCES} LINK_LIBS PUBLIC MLIRLLVMDialect MLIRIR)

target_include_directories(
  MLIRQIRUtils PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                      ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QIR/Utils/QIRUtils.h)
