# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(programs)

# Building a benchmark program as a QC program. Every backend starts from here.
add_library(MQTBenchmarkCompile STATIC Compile.cpp)
target_link_libraries(MQTBenchmarkCompile PUBLIC MQTBenchmarkPrograms MQTCompilerPipeline)
mqt_mlir_target_use_project_options(MQTBenchmarkCompile)
llvm_update_compile_flags(MQTBenchmarkCompile)

add_subdirectory(jeff)
