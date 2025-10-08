# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBRARIES ${dialect_libs} MQT::CoreIR)
add_compile_options(-fexceptions)

file(GLOB_RECURSE TRANSFORMS_SOURCES *.cpp)

add_mlir_library(MLIRMQTOptTransforms ${TRANSFORMS_SOURCES} LINK_LIBS ${LIBRARIES} DEPENDS
                 MLIRMQTOptTransformsIncGen)

# collect header files
file(GLOB_RECURSE TRANSFORMS_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/MQTOpt/Transforms/*.h)
file(GLOB_RECURSE TRANSFORMS_HEADERS_BUILD
     ${MQT_MLIR_BUILD_INCLUDE_DIR}/mlir/Dialect/MQTOpt/Transforms/*.inc)

# add public headers using file sets
target_sources(
  MLIRMQTOptTransforms
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
