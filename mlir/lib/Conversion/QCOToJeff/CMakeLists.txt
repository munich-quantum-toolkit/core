# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
  MLIRQCOToJeff
  ${CONVERSION_SOURCES}
  DEPENDS
  QCOToJeffIncGen
  LINK_LIBS
  MLIRDialect
  MLIRJeff
  MLIRQCODialect
  MLIRTransforms
  DISABLE_INSTALL)

target_include_directories(MLIRQCOToJeff PRIVATE ${jeff-mlir_SOURCE_DIR}/include
                                                 ${jeff-mlir_BINARY_DIR}/include)

if(TARGET MLIRQCOToJeff)
  target_compile_definitions(MLIRQCOToJeff PRIVATE MQT_CORE_VERSION="${MQT_CORE_VERSION}")
endif()

if(TARGET obj.MLIRQCOToJeff)
  target_compile_definitions(obj.MLIRQCOToJeff PRIVATE MQT_CORE_VERSION="${MQT_CORE_VERSION}")
endif()
