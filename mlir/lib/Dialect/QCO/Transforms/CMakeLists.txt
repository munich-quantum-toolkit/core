# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBRARIES ${dialect_libs} MQT::CoreIR)
add_compile_options(-fexceptions)

message(STATUS "MLIR_DIALECT_LIBS contains: ${dialect_libs}")

file(GLOB_RECURSE TRANSFORMS_SOURCES *.cpp)

add_mlir_library(MLIRQCOTransforms ${TRANSFORMS_SOURCES} LINK_LIBS ${LIBRARIES} DEPENDS
                 MLIRQCOTransformsIncGen)

# collect header files
file(GLOB_RECURSE TRANSFORMS_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/QCO/Transforms/*.h)
file(GLOB_RECURSE TRANSFORMS_HEADERS_BUILD
     ${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Dialect/QCO/Transforms/*.inc)

# add public headers using file sets
target_sources(
  MLIRQCOTransforms
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_SOURCE_INCLUDE_DIR}
         FILES
         ${TRANSFORMS_HEADERS_SOURCE}
         FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_BUILD_INCLUDE_DIR}
         FILES
         ${TRANSFORMS_HEADERS_BUILD})
