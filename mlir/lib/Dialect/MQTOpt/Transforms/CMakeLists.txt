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

file(GLOB TRANSFORMS_SOURCES *.cpp)

add_mlir_library(
  MLIRMQTOptTransforms
  DISABLE_INSTALL
  ${TRANSFORMS_SOURCES}
  LINK_LIBS
  ${LIBRARIES}
  DEPENDS
  MLIRMQTOptTransformsIncGen)

# collect header files
file(GLOB_RECURSE TRANSFORMS_HEADERS
     ${MQT_MLIR_INCLUDE_BUILD_DIR}/mlir/Dialect/MQTOpt/Transforms/*.h)

# add public headers using file sets
target_sources(MLIRMQTOptTransforms PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_INCLUDE_BUILD_DIR}
                                           FILES ${TRANSFORMS_HEADERS})
