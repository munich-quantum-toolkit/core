# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(target_name mqt-core-mlir-unittest-optimizations)
add_executable(
  ${target_name}
  test_qco_hadamard_lifting.cpp test_qco_measurement_lifting.cpp
  test_qco_merge_single_qubit_rotation.cpp test_qco_replace_classical_controls.cpp
  test_qco_reuse_qubits.cpp test_quantum_loop_unroll.cpp)

target_link_libraries(
  ${target_name}
  PRIVATE GTest::gtest_main
          MLIRQCOProgramBuilder
          MLIRQCOTransforms
          MLIRQCOUtils
          MLIRParser
          MLIRIR
          MLIRPass
          MLIRSupport
          LLVMSupport
          MLIRSupportMQT)

mqt_mlir_configure_unittest_target(${target_name})

gtest_discover_tests(${target_name} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
