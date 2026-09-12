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
# The existing JSON parser APIs can throw through this target.
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)
mqt_mlir_target_use_project_options(MQTBenchmarkGenerate)
unset(LLVM_REQUIRES_EH)
unset(LLVM_REQUIRES_RTTI)

# The executable that generates and evaluates configured benchmarks. LLVM option classes use the
# SDK's RTTI setting; exception handlers need RTTI on macOS.
add_library(MQTBenchmarkCommandLine OBJECT Main.cpp)
mqt_mlir_target_use_project_options(MQTBenchmarkCommandLine)
add_executable(mqt-core-bench MQTCoreBench.cpp)
if(WIN32)
  # Keep the executable import library distinct from the benchmark library.
  set_target_properties(mqt-core-bench PROPERTIES ARCHIVE_OUTPUT_NAME mqt-core-bench-executable)
endif()
target_link_libraries(mqt-core-bench PRIVATE MQTBenchmarkCommandLine MQT::CoreBenchGenerate
                                             MLIRSupport)
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)
mqt_mlir_target_use_project_options(mqt-core-bench)
unset(LLVM_REQUIRES_EH)
unset(LLVM_REQUIRES_RTTI)
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
