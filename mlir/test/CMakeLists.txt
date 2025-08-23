# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(MQT_CORE_MLIR_TEST_DEPENDS FileCheck count not split-file quantum-opt)

add_custom_target(mqt-core-mlir-lit-test-build-only DEPENDS ${MQT_CORE_MLIR_TEST_DEPENDS})
set_target_properties(mqt-core-mlir-lit-test-build-only PROPERTIES FOLDER "Tests")

add_lit_testsuite(mqt-core-mlir-lit-test "Running the MQT MLIR tests" ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS ${MQT_CORE_MLIR_TEST_DEPENDS})
set_target_properties(mqt-core-mlir-lit-test PROPERTIES FOLDER "Tests")
