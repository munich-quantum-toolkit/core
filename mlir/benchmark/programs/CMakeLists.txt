# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# The benchmark programs, one file per program. The programs build QC dialect modules and carry no
# format-specific code, so any backend can reuse them.
add_library(
  MQTBenchmarkPrograms STATIC
  BlockEncoding.cpp
  FanOut.cpp
  GHZLinear.cpp
  GHZStar.cpp
  Grover.cpp
  IQFT.cpp
  IQPE.cpp
  Multiplexer.cpp
  QAOA.cpp
  QFT.cpp
  QFTAdderClassical.cpp
  QFTAdderQuantum.cpp
  QPE.cpp
  Registry.cpp
  Teleportation.cpp
  ToffoliHeavy.cpp
  VQEAnsatz.cpp)
target_link_libraries(MQTBenchmarkPrograms PUBLIC MLIRQCProgramBuilder MLIRArithDialect)
mqt_mlir_target_use_project_options(MQTBenchmarkPrograms)
llvm_update_compile_flags(MQTBenchmarkPrograms)
