# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-benchmarks-bindings)
  add_mqt_python_binding(
    CORE
    ${MQT_CORE_TARGET_NAME}-benchmarks-bindings
    register_benchmarks.cpp
    MODULE_NAME
    benchmarks
    INSTALL_DIR
    .
    LINK_LIBS
    MQT::CoreBenchmarks)

  # install the Python stub file in editable mode for better IDE support
  if(SKBUILD_STATE STREQUAL "editable")
    install(
      FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/benchmarks.pyi
      DESTINATION .
      COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
  endif()
endif()
