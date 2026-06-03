# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(poc-test-pipeline test_pipeline.cpp)

target_link_libraries(
  poc-test-pipeline
  PRIVATE MLIRQCDialect
          MLIRQCODialect
          MLIRQTensorDialect
          MLIRArithDialect
          MLIRFuncDialect
          MLIRMemRefDialect
          MLIRSCFDialect
          MLIRQCTranslation
          MLIRQCToQCO
          MLIRSupportMQT
          MLIRPass
          MQT::CoreIR
          MQT::CoreQASM)

mqt_mlir_target_use_project_options(poc-test-pipeline)
