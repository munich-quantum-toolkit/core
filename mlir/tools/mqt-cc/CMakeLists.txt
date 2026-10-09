# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Build the compiler driver executable
add_mlir_tool(mqt-cc mqt-cc.cpp DEPENDS MQTCompilerPipeline MQTCompilerQDMIAdapter)
if(BUILD_MQT_CORE_TESTS OR MQT_CORE_INSTALL)
  # Installed LLVM SDKs may disable tools; Core's tests and installation require this executable.
  set_property(TARGET mqt-cc PROPERTY EXCLUDE_FROM_ALL FALSE)
endif()
llvm_map_components_to_libnames(llvm_native_libs bitwriter)
target_link_libraries(
  mqt-cc
  PRIVATE MQTCompilerQDMIAdapter
          MQTCompilerPipeline
          MLIRParser
          MLIRSupport
          MLIRQCOpenQASMTranslation
          MLIRJeffTranslation
          MLIRJeffToQCO
          MLIRBytecodeWriter
          MLIRQIRUtils
          MLIRTargetLLVMIRExport
          ${llvm_native_libs})

mqt_mlir_target_use_project_options(mqt-cc)

mqt_get_qdmi_device_targets(qdmi_device_targets)
if(qdmi_device_targets)
  mqt_copy_qdmi_runtime(mqt-cc ${qdmi_device_targets})
endif()

mlir_check_all_link_libraries(mqt-cc)

if(MQT_CORE_INSTALL)
  install(
    TARGETS mqt-cc
    EXPORT mqt-core-runtime-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
endif()
