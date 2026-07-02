# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-mlir-bindings)
  # collect source files
  file(GLOB_RECURSE MLIR_BINDINGS_SOURCES **.cpp)

  # declare the Python module
  add_mqt_python_binding_nanobind(
    CORE
    ${MQT_CORE_TARGET_NAME}-mlir-bindings
    ${MLIR_BINDINGS_SOURCES}
    MODULE_NAME
    _mlir
    INSTALL_DIR
    ./mlir
    LINK_LIBS
    MQTCoreCAPI
    MLIRIR)

  # install the Python stub files in editable mode for better IDE support
  if(SKBUILD_STATE STREQUAL "editable")
    file(GLOB_RECURSE MLIR_PYI_FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/mlir/*.pyi)
    install(
      FILES ${MLIR_PYI_FILES}
      DESTINATION ./mlir
      COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
  endif()
endif()
