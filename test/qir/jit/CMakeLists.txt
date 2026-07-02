# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreQIRJIT)
  set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qir-jit-test)

  # Get the LLVM native target libraries
  llvm_map_components_to_libnames(llvm_native_libs_for_test asmparser core irreader support)

  package_add_test(${TARGET_NAME} MQT::CoreQIRJIT test_ir_rewriter.cpp test_jit_session.cpp)
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQIRRuntime mqt-core-qir-test-utils
                                               ${llvm_native_libs_for_test})

  target_compile_definitions(${TARGET_NAME}
                             PRIVATE QIR_FILES_DIR="${PROJECT_SOURCE_DIR}/test/circuits")
endif()
