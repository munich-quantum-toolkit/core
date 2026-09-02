# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(target_name mqt-core-mlir-benchmark-mapping)
add_executable(${target_name} benchmark_mapping.cpp)

target_compile_options(${target_name} PRIVATE -fno-rtti)
target_link_libraries(
  ${target_name}
  PRIVATE MLIRParser
          MQTCompilerTarget
          MLIRQCOProgramBuilder
          MLIRQTensorUtils
          MLIRQCOTransforms
          MLIRSupportMQT
          MLIRPass
          MLIRSupport)

set(target_name mqt-core-mlir-benchmark-generator)
add_executable(${target_name} benchmark_generator.cpp)

target_compile_options(${target_name} PRIVATE -fno-rtti)
target_link_libraries(
  ${target_name}
  PRIVATE MLIRParser
          MQTCompilerTarget
          MQTCompilerPipeline
          MLIRQCOProgramBuilder
          MLIRQTensorUtils
          MLIRQCOTransforms
          MLIRSupportMQT
          MLIRPass
          MLIRSupport)