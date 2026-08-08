# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-dd-eval eval_dd_package.cpp)
target_link_libraries(
  mqt-core-dd-eval PRIVATE MQT::CoreDD MQT::CoreAlgorithms MQT::CoreCircuitOptimizer
                           MQT::ProjectOptions MQT::ProjectWarnings)

if(BUILD_MQT_CORE_MLIR)
  add_executable(mqt-core-mlir-mapping-eval mlir_mapping.cpp)
  target_link_libraries(
    mqt-core-mlir-mapping-eval PRIVATE MLIRParser MLIRQCOProgramBuilder MLIRQTensorUtils
                                       MLIRQCOTransforms MQT::ProjectOptions MQT::ProjectWarnings)
endif()
