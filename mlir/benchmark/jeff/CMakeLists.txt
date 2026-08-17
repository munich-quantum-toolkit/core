# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# The lowering of a benchmark program to `jeff`
add_library(MQTJeffBenchmarks STATIC Generate.cpp)
target_include_directories(MQTJeffBenchmarks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(MQTJeffBenchmarks PUBLIC MQTBenchmarkPrograms MQTCompilerPipeline)
mqt_mlir_target_use_project_options(MQTJeffBenchmarks)
llvm_update_compile_flags(MQTJeffBenchmarks)

# The generator executable
add_executable(mqt-jeff-benchmarks MQTJeffBenchmarks.cpp)
target_link_libraries(mqt-jeff-benchmarks PRIVATE MQTJeffBenchmarks MLIRSupport)
mqt_mlir_target_use_project_options(mqt-jeff-benchmarks)
llvm_update_compile_flags(mqt-jeff-benchmarks)
mlir_check_all_link_libraries(mqt-jeff-benchmarks)
