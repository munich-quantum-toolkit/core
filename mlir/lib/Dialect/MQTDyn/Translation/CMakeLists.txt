# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(MLIRMQTDynTranslation ImportQuantumComputation.cpp LINK_LIBS MLIRMQTDyn
                 MLIRFuncDialect MQT::CoreIR)

target_compile_options(MLIRMQTDynTranslation PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fexceptions>)

# collect header files
file(GLOB_RECURSE TRANSLATION_HEADERS_SOURCE
     ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Dialect/MQTDyn/Translation/*.h)

# add public headers using file sets
target_sources(
  MLIRMQTDynTranslation PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                               ${TRANSLATION_HEADERS_SOURCE})
