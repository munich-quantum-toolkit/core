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
add_executable(mqt-core-bench MQTCoreBench.cpp)
if(WIN32)
  # Keep the executable import library distinct from the benchmark library.
  set_target_properties(mqt-core-bench PROPERTIES ARCHIVE_OUTPUT_NAME mqt-core-bench-executable)
endif()
target_link_libraries(mqt-core-bench PRIVATE MQT::CoreBenchGenerate MLIRSupport)
set(LLVM_REQUIRES_EH ON)
mqt_mlir_target_use_project_options(mqt-core-bench)
unset(LLVM_REQUIRES_EH)
mlir_check_all_link_libraries(mqt-core-bench)

if(MQT_CORE_INSTALL)
  include(GNUInstallDirs)
  install(TARGETS mqt-core-bench RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
                                         COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
endif()
