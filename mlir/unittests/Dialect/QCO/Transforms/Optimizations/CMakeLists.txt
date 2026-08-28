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
  test_qco_auxiliary_qubit_hoisting.cpp
  test_qco_context_sensitive_specialization.cpp
  test_qco_function_boundary_commutation.cpp
  test_qco_hadamard_lifting.cpp
  test_qco_measurement_lifting.cpp
  test_qco_merge_single_qubit_rotation.cpp
  test_qco_pauli_twirling.cpp
  test_qco_quantum_argument_promotion.cpp
  test_qco_remove_dead_gates.cpp
  test_qco_replace_classical_controls.cpp
  test_qco_reuse_qubits.cpp
  test_quantum_loop_unroll.cpp)

target_link_libraries(
  ${target_name}
  PRIVATE GTest::gtest_main
          MLIRControlFlowDialect
          MLIRQCODDFunctionality
          MLIRQCOProgramBuilder
          MLIRQCOPrograms
          MLIRQCOTransforms
          MLIRQCOUtils
          MLIRParser
          MLIRIR
          MLIRPass
          MLIRSupport
          LLVMSupport
          MLIRSupportMQT)

mqt_mlir_configure_unittest_target(${target_name} REQUIRES_EH)

gtest_discover_tests(${target_name} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
