# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Build the compiler driver executable
set(native_compiler ${MQT_CORE_TARGET_NAME}-native-compiler)
if(TARGET ${native_compiler})
  target_sources(${native_compiler} PRIVATE mqt-cc.cpp)
  set_source_files_properties(
    mqt-cc.cpp
    TARGET_DIRECTORY ${native_compiler}
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
  target_compile_definitions(${native_compiler} PRIVATE MQT_CORE_SHARED_COMPILER_CLI)
  if(APPLE)
    target_link_options(${native_compiler} PRIVATE "LINKER:-exported_symbol,_mqtCoreRunCompiler")
  endif()
  add_mlir_tool(mqt-cc CompilerMain.cpp PARTIAL_SOURCES_INTENDED)
  target_link_libraries(mqt-cc PRIVATE ${native_compiler})
  set_target_properties(
    mqt-cc
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
  set(compiler_cli ${native_compiler})
else()
  add_mlir_tool(mqt-cc mqt-cc.cpp PARTIAL_SOURCES_INTENDED DEPENDS MQTCompilerPipeline
                MQTCompilerQDMIAdapter)
  set(compiler_cli mqt-cc)
endif()
if(BUILD_MQT_CORE_TESTS OR MQT_CORE_INSTALL)
  # Installed LLVM SDKs may disable tools; Core's tests and installation require this executable.
  set_property(TARGET mqt-cc PROPERTY EXCLUDE_FROM_ALL FALSE)
endif()
target_link_libraries(
  ${compiler_cli}
  PRIVATE
    MQTCompilerQDMIAdapter
    MQTCompilerPipeline
    MLIRParser
    MLIRSupport
    MLIRQCOpenQASMTranslation
    MLIRJeffTranslation
    MLIRJeffToQCO
    MLIRBytecodeWriter
    MLIRQIRTranslation)

mqt_mlir_target_use_project_options(${compiler_cli})
mqt_copy_qdmi_runtime(mqt-cc)

mlir_check_all_link_libraries(mqt-cc)

if(MQT_CORE_INSTALL)
  install(
    TARGETS mqt-cc
    EXPORT mqt-core-runtime-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
endif()
