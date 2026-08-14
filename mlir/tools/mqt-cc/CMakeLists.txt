# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Build the compiler driver executable
add_mlir_tool(mqt-cc mqt-cc.cpp DEPENDS MQTCompilerPipeline MQTCompilerQDMIAdapter SUPPORT_PLUGINS)
if(BUILD_MQT_CORE_TESTS)
  # add_mlir_tool excludes tools when LLVM_BUILD_TOOLS is disabled. The driver tests require mqt-cc
  # in test-enabled default builds.
  set_property(TARGET mqt-cc PROPERTY EXCLUDE_FROM_ALL FALSE)
endif()
llvm_map_components_to_libnames(llvm_native_libs bitwriter)
target_link_libraries(
  mqt-cc
  PRIVATE MQTCompilerQDMIAdapter
          MQTCompilerPipeline
          MLIRParser
          MLIRSupport
          MLIRQCTranslation
          MLIRJeffTranslation
          MLIRJeffToQCO
          MLIRBytecodeWriter
          MLIRMathDialect
          MLIRQIRUtils
          MLIRTargetLLVMIRExport
          MLIRBuiltinToLLVMIRTranslation
          MLIRLLVMToLLVMIRTranslation
          ${llvm_native_libs})

mqt_mlir_target_use_project_options(mqt-cc)

mqt_get_qdmi_device_targets(qdmi_device_targets)
if(qdmi_device_targets)
  mqt_copy_qdmi_runtime(mqt-cc ${qdmi_device_targets})
endif()

llvm_update_compile_flags(mqt-cc)
mlir_check_all_link_libraries(mqt-cc)
export_executable_symbols_for_plugins(mqt-cc)
