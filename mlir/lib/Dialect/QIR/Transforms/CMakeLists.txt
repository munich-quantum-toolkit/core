# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB_RECURSE PASSES_SOURCES *.cpp)

add_mlir_library(
        MLIRQIRTransforms
        ${PASSES_SOURCES}
        LINK_LIBS
        PRIVATE
        MLIRLLVMDialect
        MLIRQIRUtils
        DEPENDS
        MLIRQIRTransformsIncGen)

mqt_mlir_target_use_project_options(MLIRQIRTransforms)

# collect header files
file(GLOB_RECURSE PASSES_HEADERS_SOURCE
        ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QIR/Transforms/*.h)
file(GLOB_RECURSE PASSES_HEADERS_BUILD
        ${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Dialect/QIR/Transforms/*.inc)

# add public headers using file sets
target_sources(
        MLIRQIRTransforms
        PUBLIC FILE_SET
        HEADERS
        BASE_DIRS
        ${MQT_MLIR_SOURCE_INCLUDE_DIR}
        FILES
        ${PASSES_HEADERS_SOURCE}
        FILE_SET
        HEADERS
        BASE_DIRS
        ${MQT_MLIR_BUILD_INCLUDE_DIR}
        FILES
        ${PASSES_HEADERS_BUILD})
