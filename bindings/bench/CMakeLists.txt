# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-bench-bindings)
  add_mqt_python_binding(
    CORE
    ${MQT_CORE_TARGET_NAME}-bench-bindings
    register_bench.cpp
    MODULE_NAME
    bench
    INSTALL_DIR
    .
    LINK_LIBS
    MQT::CoreBench)

  # install the Python stub file in editable mode for better IDE support
  if(SKBUILD_STATE STREQUAL "editable")
    install(
      FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/bench.pyi
      DESTINATION .
      COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
  endif()
endif()
