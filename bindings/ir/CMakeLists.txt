# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-ir-bindings)
  # collect source files
  file(GLOB_RECURSE IR_SOURCES **.cpp)

  # declare the Python module
  add_mqt_python_binding_nanobind(
    CORE
    ${MQT_CORE_TARGET_NAME}-ir-bindings
    ${IR_SOURCES}
    MODULE_NAME
    ir
    INSTALL_DIR
    .
    LINK_LIBS
    MQT::CoreIR
    MQT::CoreQASM)

  nanobind_add_stub(
    ${MQT_CORE_TARGET_NAME}-ir-bindings_stub
    MODULE
    mqt.core.ir
    RECURSIVE
    OUTPUT_PATH
    ${PROJECT_SOURCE_DIR}/python/mqt/core/
    OUTPUT
    ${PROJECT_SOURCE_DIR}/python/mqt/core/ir/__init__.pyi
    ${PROJECT_SOURCE_DIR}/python/mqt/core/ir/operations.pyi
    ${PROJECT_SOURCE_DIR}/python/mqt/core/ir/registers.pyi
    ${PROJECT_SOURCE_DIR}/python/mqt/core/ir/symbolic.pyi
    PYTHON_PATH
    ${PROJECT_SOURCE_DIR}/python
    MARKER_FILE
    py.typed)
endif()
