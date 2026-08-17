# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# The benchmark programs, one file per program, and the lowering to `jeff`
add_library(
  MQTJeffBenchmarkPrograms STATIC
  Generate.cpp
  programs/GHZLinear.cpp
  programs/GHZStar.cpp
  programs/Grover.cpp
  programs/IQFT.cpp
  programs/IQPE.cpp
  programs/Multiplexer.cpp
  programs/QFT.cpp
  programs/QPE.cpp
  programs/Registry.cpp
  programs/Teleportation.cpp)
target_include_directories(MQTJeffBenchmarkPrograms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(MQTJeffBenchmarkPrograms PUBLIC MLIRQCProgramBuilder MLIRArithDialect
                                                      MQTCompilerPipeline)
mqt_mlir_target_use_project_options(MQTJeffBenchmarkPrograms)
llvm_update_compile_flags(MQTJeffBenchmarkPrograms)

# The generator executable
add_executable(mqt-jeff-benchmarks MQTJeffBenchmarks.cpp)
target_link_libraries(mqt-jeff-benchmarks PRIVATE MQTJeffBenchmarkPrograms MLIRSupport)
mqt_mlir_target_use_project_options(mqt-jeff-benchmarks)
llvm_update_compile_flags(mqt-jeff-benchmarks)
mlir_check_all_link_libraries(mqt-jeff-benchmarks)
