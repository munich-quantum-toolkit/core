# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(CatalystQuantumToMQTOpt CatalystQuantumToMQTOpt.cpp LINK_LIBS MLIRMQTOpt DEPENDS
                 CatalystQuantumToMQTOptIncGen)

target_compile_features(CatalystQuantumToMQTOpt PUBLIC cxx_std_20)
target_compile_options(CatalystQuantumToMQTOpt PUBLIC -fexceptions)

file(GLOB_RECURSE CONVERSION_HEADERS_SOURCE
     "${MQT_MLIR_PLUGIN_SOURCE_INCLUDE_DIR}/mlir/Conversion/CatalystQuantumToMQTOpt/*.h")
file(GLOB_RECURSE CONVERSION_HEADERS_BUILD
     "${MQT_MLIR_PLUGIN_BUILD_INCLUDE_DIR}/mlir/Conversion/CatalystQuantumToMQTOpt/*.inc")

# add public headers using file sets
target_sources(
  CatalystQuantumToMQTOpt
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_PLUGIN_SOURCE_INCLUDE_DIR}
         FILES
         ${CONVERSION_HEADERS_SOURCE}
         FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_PLUGIN_BUILD_INCLUDE_DIR}
         FILES
         ${CONVERSION_HEADERS_BUILD})
