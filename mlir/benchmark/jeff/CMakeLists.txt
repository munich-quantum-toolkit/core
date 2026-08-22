# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# The executable that generates and evaluates configured benchmarks.
add_executable(mqt-core-bench MQTJeffBenchmarks.cpp)
target_link_libraries(mqt-core-bench PRIVATE MQT::CoreBenchmarks MQTBenchmarkGenerate MLIRSupport)
set(LLVM_REQUIRES_EH ON)
mqt_mlir_target_use_project_options(mqt-core-bench)
unset(LLVM_REQUIRES_EH)
mlir_check_all_link_libraries(mqt-core-bench)

if(MQT_CORE_INSTALL)
  if(SKBUILD)
    set(mqt_core_bench_install_dir "${SKBUILD_SCRIPTS_DIR}")
  else()
    include(GNUInstallDirs)
    set(mqt_core_bench_install_dir "${CMAKE_INSTALL_BINDIR}")
  endif()
  install(TARGETS mqt-core-bench RUNTIME DESTINATION "${mqt_core_bench_install_dir}"
                                         COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
endif()
