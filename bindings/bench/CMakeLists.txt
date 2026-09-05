# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-bench-bindings)
  set(BENCH_SOURCES
      register_bench.cpp
      register_bv.cpp
      register_ghz.cpp
      register_grover.cpp
      register_multiplexer.cpp
      register_qft.cpp
      register_qpe.cpp
      register_teleportation.cpp)

  add_mqt_python_binding(
    CORE
    ${MQT_CORE_TARGET_NAME}-bench-bindings
    ${BENCH_SOURCES}
    MODULE_NAME
    bench
    INSTALL_DIR
    .
    LINK_LIBS
    MQT::CoreBench)

  # install the Python stub files in editable mode for better IDE support
  if(SKBUILD_STATE STREQUAL "editable")
    file(GLOB_RECURSE BENCH_PYI_FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/bench/*.pyi)
    install(
      FILES ${BENCH_PYI_FILES}
      DESTINATION ./bench
      COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
  endif()
endif()
