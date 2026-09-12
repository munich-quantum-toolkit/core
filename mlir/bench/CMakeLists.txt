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
add_library(MQT::CoreBenchGenerate ALIAS MQTBenchmarkGenerate)
target_link_libraries(MQTBenchmarkGenerate PUBLIC MQTBenchmarkPrograms MQTCompilerPipeline)
mqt_mlir_target_use_project_options(MQTBenchmarkGenerate)

# The executable that generates and evaluates configured benchmarks.
add_executable(mqt-core-bench Main.cpp MQTCoreBench.cpp)
if(WIN32)
  # Keep the executable import library distinct from the benchmark library.
  set_target_properties(mqt-core-bench PROPERTIES ARCHIVE_OUTPUT_NAME mqt-core-bench-executable)
endif()
target_link_libraries(mqt-core-bench PRIVATE MQT::CoreBenchGenerate MLIRSupport)
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)
mqt_mlir_target_use_project_options(mqt-core-bench)
unset(LLVM_REQUIRES_EH)
unset(LLVM_REQUIRES_RTTI)
if(NOT LLVM_ENABLE_RTTI)
  # LLVM command-line option types need LLVM's RTTI setting. The exception handler needs RTTI to
  # catch exceptions from shared libraries on macOS.
  set_source_files_properties(
    MQTCoreBench.cpp
    PROPERTIES COMPILE_OPTIONS "${LLVM_CXXFLAGS_RTTI_DISABLE}"
               SKIP_UNITY_BUILD_INCLUSION ON
               SKIP_PRECOMPILE_HEADERS ON)
endif()
mlir_check_all_link_libraries(mqt-core-bench)
if(UNIX AND NOT APPLE)
  target_link_options(
    mqt-core-bench PRIVATE
    "$<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>,$<CONFIG:RelWithDebInfo>>:LINKER:--gc-sections>")
endif()

if(MQT_CORE_INSTALL)
  include(GNUInstallDirs)
  install(TARGETS mqt-core-bench RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
                                         COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
endif()
