# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME "${MQT_CORE_TARGET_NAME}-na-bindings")

if(NOT TARGET ${TARGET_NAME})
  # collect source files
  file(GLOB_RECURSE SOURCES *.cpp)

  # declare the Python module
  add_mqt_python_binding_nanobind(
    CORE
    ${TARGET_NAME}
    ${SOURCES}
    MODULE_NAME
    na
    INSTALL_DIR
    .
    LINK_LIBS
    MQT::CoreNAFoMaC)

  # install the Python stub file in editable mode for better IDE support
  if(SKBUILD_STATE STREQUAL "editable")
    file(GLOB_RECURSE NA_PYI_FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/na/*.pyi)
    install(
      FILES ${NA_PYI_FILES}
      DESTINATION ./na
      COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
  endif()
endif()
