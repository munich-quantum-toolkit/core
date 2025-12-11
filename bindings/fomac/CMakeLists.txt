# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME "${MQT_CORE_TARGET_NAME}-fomac-bindings")

if(NOT TARGET ${TARGET_NAME})
  # collect source files
  file(GLOB_RECURSE SOURCES **.cpp)

  # declare the Python module
  add_mqt_python_binding_nanobind(
    CORE
    ${TARGET_NAME}
    ${SOURCES}
    MODULE_NAME
    fomac
    INSTALL_DIR
    .
    LINK_LIBS
    MQT::CoreFoMaC)

  nanobind_add_stub(
    ${TARGET_NAME}_stub
    INSTALL_TIME
    MODULE
    mqt.core.fomac
    OUTPUT
    ${PROJECT_SOURCE_DIR}/python/mqt/core/fomac.pyi
    PYTHON_PATH
    ${PROJECT_SOURCE_DIR}/python
    MARKER_FILE
    py.typed
    INCLUDE_PRIVATE)
endif()
