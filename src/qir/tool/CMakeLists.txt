# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qir-runner)

if(NOT TARGET ${TARGET_NAME})
  add_llvm_tool(${TARGET_NAME} App.cpp DEPENDS intrinsics_gen EXPORT_SYMBOLS)

  # Add link libraries
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQIRBackend ${llvm_libs})

  # Set versioning information
  set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${PROJECT_VERSION} EXPORT_NAME
                                                                             CoreQIRRunner)
  # place in the bin directory
  set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  "${CMAKE_BINARY_DIR}/bin")

  # Set c++ standard
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)

  # Make version available
  target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_VERSION="${MQT_CORE_VERSION}")
endif()
