# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qir-runner)

if(NOT TARGET ${TARGET_NAME})
  add_llvm_tool(${TARGET_NAME} Runner.cpp DEPENDS intrinsics_gen EXPORT_SYMBOLS)

  # Get the native target libraries
  llvm_map_components_to_libnames(llvm_native_libs native)

  # Add link libraries
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQIRRuntime LLVMOrcDebugging
                                               ${llvm_native_libs})

  # Set versioning information
  set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${PROJECT_VERSION} EXPORT_NAME
                                                                             CoreQIRRunner)
  # Set the appropriate output directories
  set_target_properties(
    ${TARGET_NAME}
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

  # Set c++ standard
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)

  # Make version available
  target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_VERSION="${MQT_CORE_VERSION}")

  # Create an alias for the target
  add_executable(MQT::CoreQIRRunner ALIAS ${TARGET_NAME})
endif()
