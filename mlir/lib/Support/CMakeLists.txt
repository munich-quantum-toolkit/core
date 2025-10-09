# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT

add_mlir_library(
  MLIRSupportMQT
  PrettyPrinting.cpp
  TestUtils.cpp
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Support
  LINK_LIBS
  PUBLIC
  MLIRSupport
  MLIRIR)

# add library alias
add_library(MQT::MLIRSupport ALIAS MLIRSupportMQT)

# collect header files
file(GLOB_RECURSE SUPPORT_HEADERS_SOURCE "${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Support/*.h")
target_sources(MLIRSupportMQT PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
                                     FILES ${SUPPORT_HEADERS_SOURCE})
