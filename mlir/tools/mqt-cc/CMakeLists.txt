# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Build the compiler driver executable
add_mlir_tool(mqt-cc mqt-cc.cpp DEPENDS MQTCompilerPipeline SUPPORT_PLUGINS)
llvm_map_components_to_libnames(llvm_native_libs bitwriter)
target_link_libraries(
  mqt-cc
  PRIVATE MQTCompilerPipeline
          MLIRParser
          MLIRSupport
          MLIRQCTranslation
        MLIRJeffTranslation
        MLIRJeffToQCO
          MLIRBytecodeWriter
          MLIRTargetLLVMIRExport
          MLIRBuiltinToLLVMIRTranslation
          MLIRLLVMToLLVMIRTranslation
          ${llvm_native_libs})

mqt_mlir_target_use_project_options(mqt-cc)
llvm_update_compile_flags(mqt-cc)
mlir_check_all_link_libraries(mqt-cc)
export_executable_symbols_for_plugins(mqt-cc)
