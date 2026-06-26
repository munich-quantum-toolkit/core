# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME "${MQT_CORE_TARGET_NAME}-mlir-bindings")

if(NOT TARGET ${TARGET_NAME})
  # collect source files
  file(GLOB_RECURSE SOURCES *.cpp)

  # declare the Python module
  add_mqt_python_binding_nanobind(
    CORE
    ${TARGET_NAME}
    ${SOURCES}
    MODULE_NAME
    mlir
    INSTALL_DIR
    .
    LINK_LIBS
    MQTCompilerPipeline
    MLIRQCTranslation
    MLIRParser
    MLIRJeffTranslation
    MLIRJeffToQCO
    MLIRQCOToQC
    MQT::MLIRSupport
    MQT::CoreIR)

  # install the Python stub file in editable mode for better IDE support
  if(SKBUILD_STATE STREQUAL "editable")
    if(EXISTS "${PROJECT_SOURCE_DIR}/python/mqt/core/mlir.pyi")
      install(
        FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/mlir.pyi
        DESTINATION .
        COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
    endif()
  endif()
endif()
