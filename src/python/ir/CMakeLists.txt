# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ir)
  # collect source files
  file(GLOB_RECURSE IR_SOURCES **.cpp)

  # declare the Python module
  pybind11_add_module(
    # Name of the extension
    ir
    # Prefer thin LTO if available
    THIN_LTO
    # Optimize the bindings for size
    OPT_SIZE
    # Source code goes here
    ${MQT_CORE_INCLUDE_BUILD_DIR}/python/pybind11.hpp
    ${IR_SOURCES})
  target_link_libraries(ir PRIVATE MQT::CoreIR MQT::CoreQASM MQT::ProjectOptions
                                   MQT::ProjectWarnings)

  # Install directive for scikit-build-core
  install(
    TARGETS ir
    DESTINATION .
    COMPONENT ${MQT_CORE_TARGET_NAME}_Python)

  # Install the Python stub files in editable mode for better IDE support
  if(SKBUILD_STATE STREQUAL "editable")
    file(GLOB_RECURSE IR_PYI_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../mqt/core/ir/*.pyi)
    install(
      FILES ${IR_PYI_FILES}
      DESTINATION ./ir
      COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
  endif()
endif()
