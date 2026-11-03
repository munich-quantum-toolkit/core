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
set(native_compiler ${MQT_CORE_TARGET_NAME}-native-compiler)
if(TARGET ${native_compiler})
  target_sources(${native_compiler} PRIVATE MQTCoreBench.cpp)
  target_compile_definitions(${native_compiler} PRIVATE MQT_CORE_SHARED_BENCHMARK)
  target_link_libraries(${native_compiler} PRIVATE MQT::CoreBenchGenerate MLIRSupport)
  if(APPLE)
    target_link_options(${native_compiler} PRIVATE "LINKER:-exported_symbol,_mqtCoreRunBench")
  endif()
  add_executable(mqt-core-bench BenchmarkMain.cpp)
  target_link_libraries(mqt-core-bench PRIVATE ${native_compiler})
  set_target_properties(
    mqt-core-bench
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
else()
  add_executable(mqt-core-bench MQTCoreBench.cpp)
  target_link_libraries(mqt-core-bench PRIVATE MQT::CoreBenchGenerate MLIRSupport)
endif()
if(WIN32)
  # Keep the executable import library distinct from the benchmark library.
  set_target_properties(mqt-core-bench PROPERTIES ARCHIVE_OUTPUT_NAME mqt-core-bench-executable)
endif()
mqt_mlir_target_use_project_options(mqt-core-bench)
mlir_check_all_link_libraries(mqt-core-bench)

if(MQT_CORE_INSTALL)
  include(GNUInstallDirs)
  install(
    TARGETS mqt-core-bench
    EXPORT mqt-core-runtime-targets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
endif()
