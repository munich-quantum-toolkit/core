# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB_RECURSE SUPPORT_CPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_mlir_library(
  MLIRSupportMQT
  ${SUPPORT_CPP}
  ADDITIONAL_HEADER_DIRS
  ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Support
  LINK_LIBS
  PRIVATE
  nlohmann_json::nlohmann_json
  PUBLIC
  MLIRSupport
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRControlFlowInterfaces
  MLIRTransformUtils
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRQCTransforms
  MLIRQIRTransforms
  MLIRQTensorTransforms
  MLIRQTensorDialect
  MQT::CoreFoMaC)

mqt_mlir_target_use_project_options(MLIRSupportMQT)

# add library alias
add_library(MQT::MLIRSupport ALIAS MLIRSupportMQT)

# collect header files
file(GLOB_RECURSE SUPPORT_HEADERS_SOURCE "${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Support/*.h")
target_sources(MLIRSupportMQT PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
                                     FILES ${SUPPORT_HEADERS_SOURCE})
