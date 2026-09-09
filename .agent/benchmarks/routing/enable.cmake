# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

include_guard(GLOBAL)

# Add the benchmark after MQT Core has defined its libraries and build options.
function(add_routing_benchmark)
  add_executable(mqt-core-mlir-benchmark-mapping EXCLUDE_FROM_ALL
                 "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/benchmark_mapping.cpp")
  target_link_libraries(mqt-core-mlir-benchmark-mapping PRIVATE MLIRQCOProgramBuilder
                                                                MLIRQCOTransforms MQTCompilerTarget)
  mqt_mlir_target_use_project_options(mqt-core-mlir-benchmark-mapping)

  set_target_properties(mqt-core-mlir-benchmark-mapping
                        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/routing-benchmark")
endfunction()

cmake_language(DEFER CALL add_routing_benchmark)
