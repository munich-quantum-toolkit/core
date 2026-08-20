# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(mqt_transforms_target mqt-core-mlir-unittests-mqt-transforms)
add_executable(${mqt_transforms_target} test_global_phase_normalization.cpp)

target_link_libraries(
  ${mqt_transforms_target}
  PRIVATE GTest::gtest_main
          MLIRArithDialect
          MLIRControlFlowDialect
          MLIRFuncDialect
          MLIRIR
          MLIRMemRefDialect
          MLIRParser
          MLIRMQTTransforms
          MLIRQCProgramBuilder
          MLIRQCToQCO
          MLIRQCDialect
          MLIRQCODDFunctionality
          MLIRExactUnitaryTest
          MLIRQCODialect
          MLIRQTensorDialect
          MLIRSupport
          MLIRSCFDialect
          LLVMSupport
          MQT::CoreDD)

mqt_mlir_configure_unittest_target(${mqt_transforms_target} REQUIRES_EH)

gtest_discover_tests(${mqt_transforms_target} PROPERTIES LABELS mqt-mlir-unittests
                                                         DISCOVERY_TIMEOUT 60)
