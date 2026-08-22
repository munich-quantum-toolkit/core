# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB PROGRAM_SOURCES *.cpp)

add_library(MQTBenchmarkPrograms STATIC ${PROGRAM_SOURCES})
target_link_libraries(MQTBenchmarkPrograms PUBLIC MLIRQCProgramBuilder MLIRArithDialect)
mqt_mlir_target_use_project_options(MQTBenchmarkPrograms)
llvm_update_compile_flags(MQTBenchmarkPrograms)
