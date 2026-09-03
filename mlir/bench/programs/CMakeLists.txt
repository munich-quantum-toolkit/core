# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_library(MQTBenchmarkPrograms STATIC BV.cpp GHZ.cpp Grover.cpp Multiplexer.cpp QFT.cpp QPE.cpp)
target_link_libraries(MQTBenchmarkPrograms PUBLIC MQT::CoreBench MLIRQCProgramBuilder
                                                  MLIRArithDialect MLIRTensorDialect)
mqt_mlir_target_use_project_options(MQTBenchmarkPrograms)
