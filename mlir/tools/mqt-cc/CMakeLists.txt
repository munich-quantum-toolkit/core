# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Build the compiler driver executable
add_mlir_tool(mqt-cc mqt-cc.cpp DEPENDS MQTCompilerPipeline SUPPORT_PLUGINS)
target_compile_options(mqt-cc PRIVATE -fexceptions)
target_link_libraries(
  mqt-cc
  PRIVATE MQTCompilerPipeline
          # Required for parsing .mlir files
          MLIRParser
          # Required for file I/O
          MLIRSupport)
llvm_update_compile_flags(mqt-cc)
mlir_check_all_link_libraries(mqt-cc)
export_executable_symbols_for_plugins(mqt-cc)
