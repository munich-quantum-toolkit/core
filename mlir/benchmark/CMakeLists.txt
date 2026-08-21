# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(programs)

# Generating a benchmark program as a QC program. Every backend starts from here.
add_library(MQTBenchmarkGenerate STATIC Generate.cpp)
target_link_libraries(MQTBenchmarkGenerate PUBLIC MQTBenchmarkPrograms MQTCompilerPipeline)
mqt_mlir_target_use_project_options(MQTBenchmarkGenerate)
llvm_update_compile_flags(MQTBenchmarkGenerate)

add_subdirectory(jeff)
