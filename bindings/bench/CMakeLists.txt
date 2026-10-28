# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(BENCH_SOURCES
    register_bench.cpp
    register_bv.cpp
    register_fan_out.cpp
    register_ghz.cpp
    register_grover.cpp
    register_grover_weak_measurement.cpp
    register_magic_state_distillation.cpp
    register_modular_multiplier.cpp
    register_multiplexer.cpp
    register_qaoa.cpp
    register_qft.cpp
    register_qft_adder.cpp
    register_qpe.cpp
    register_repeat_until_success.cpp
    register_shor.cpp
    register_teleportation.cpp
    register_w_state.cpp)

target_sources(${MQT_CORE_TARGET_NAME}-bindings PRIVATE ${BENCH_SOURCES})
target_link_libraries(${MQT_CORE_TARGET_NAME}-bindings PRIVATE MQT::CoreBench)

# install the Python stub files in editable mode for better IDE support
if(MQT_CORE_INSTALL AND SKBUILD_STATE STREQUAL "editable")
  file(GLOB_RECURSE BENCH_PYI_FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/bench/*.pyi)
  install(FILES ${BENCH_PYI_FILES} DESTINATION ./bench COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
endif()
