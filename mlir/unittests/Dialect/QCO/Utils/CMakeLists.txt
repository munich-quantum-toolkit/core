# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(qco_utils_target mqt-core-mlir-unittest-qco-utils)
add_executable(
  ${qco_utils_target}
  test_dd_functionality.cpp test_drivers.cpp test_layout.cpp test_phase_estimation_example.cpp
  test_unitary_matrix.cpp test_wireiterator.cpp)
target_compile_definitions(
  ${qco_utils_target}
  PRIVATE
    QC_ITERATIVE_PHASE_ESTIMATION_EXAMPLE="${PROJECT_SOURCE_DIR}/mlir/examples/qc_iterative_phase_estimation.mlir"
    QC_QUANTUM_PHASE_ESTIMATION_EXAMPLE="${PROJECT_SOURCE_DIR}/mlir/examples/qc_quantum_phase_estimation.mlir"
    QASM3_ITERATIVE_PHASE_ESTIMATION_EXAMPLE="${PROJECT_SOURCE_DIR}/mlir/examples/qasm3_iterative_phase_estimation.qasm"
    QASM3_QUANTUM_PHASE_ESTIMATION_EXAMPLE="${PROJECT_SOURCE_DIR}/mlir/examples/qasm3_quantum_phase_estimation.qasm"
)
target_link_libraries(
  ${qco_utils_target}
  PRIVATE GTest::gtest_main
          MLIRParser
          MLIRQCTranslation
          MLIRQCToQCO
          MLIRQCODDFunctionality
          MLIRQCOUtils
          MLIRQCOProgramBuilder
          MLIRQCOMatrix
          MLIRSupportMQT)
mqt_mlir_configure_unittest_target(${qco_utils_target})

gtest_discover_tests(${qco_utils_target} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
