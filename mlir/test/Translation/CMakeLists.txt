# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

include(GoogleTest)
file(GLOB_RECURSE TRANSLATION_TEST_SOURCES *.cpp)
package_add_test_with_working_dir(mqt-mlir-translation-test MLIRMQTDynImportQuantumComputation
                                  ${CMAKE_CURRENT_BINARY_DIR} ${TRANSLATION_TEST_SOURCES})
target_link_libraries(mqt-mlir-translation-test PRIVATE MQT::CoreIR MLIRMQTDyn MLIR LLVM)
