# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-dd-bindings)
  # collect source files
  file(GLOB_RECURSE DD_SOURCES **.cpp)

  # declare the Python module
  add_mqt_python_binding_nanobind(
    CORE
    ${MQT_CORE_TARGET_NAME}-dd-bindings
    ${DD_SOURCES}
    MODULE_NAME
    dd
    INSTALL_DIR
    .
    LINK_LIBS
    MQT::CoreDD)

  nanobind_add_stub(
    ${MQT_CORE_TARGET_NAME}-dd-bindings_stub
    MODULE
    mqt.core.dd
    OUTPUT
    ${PROJECT_SOURCE_DIR}/python/mqt/core/dd.pyi
    PYTHON_PATH
    ${PROJECT_SOURCE_DIR}/python
    MARKER_FILE
    py.typed
    INCLUDE_PRIVATE)
endif()
