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

file(GLOB TRANSLATION_SOURCES *.cpp)

add_mlir_library(MLIRMQTDynTranslation ${TRANSLATION_SOURCES} LINK_LIBS ${LIBRARIES})

# collect header files
file(GLOB_RECURSE TRANSLATION_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/MQTDyn/Translation/*.h)

# add public headers using file sets
target_sources(
  MLIRMQTDynTranslation PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                               ${TRANSLATION_HEADERS_SOURCE})
