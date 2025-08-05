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

set(QUANTUM_CONVERT_TEST_DEPENDS FileCheck count not quantum-convert)

add_lit_testsuite(check-quantum-convert "Running the quantum-convert regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${QUANTUM_CONVERT_TEST_DEPENDS})
set_target_properties(check-quantum-convert PROPERTIES FOLDER "Tests" EXCLUDE_FROM_ALL ON)

add_lit_testsuites(QUANTUM_CONVERT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS
                   ${QUANTUM_CONVERT_TEST_DEPENDS})
