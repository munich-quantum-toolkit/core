# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(ir)
add_subdirectory(dd)
add_subdirectory(fomac)
add_subdirectory(na)

if(SKBUILD_STATE STREQUAL "editable")
  nanobind_add_stub(
    ir_stub
    INSTALL_TIME
    RECURSIVE
    MODULE
    mqt.core.ir
    OUTPUT
    ir/__init__.pyi
    ir/operations.pyi
    ir/registers.pyi
    ir/symbolic.pyi
    PYTHON_PATH
    ${CMAKE_INSTALL_PREFIX}/../..
    MARKER_FILE
    ir/py.typed
    INCLUDE_PRIVATE)

  nanobind_add_stub(
    dd_stub
    INSTALL_TIME
    MODULE
    mqt.core.dd
    OUTPUT
    dd.pyi
    PYTHON_PATH
    ${CMAKE_INSTALL_PREFIX}/../..
    MARKER_FILE
    py.typed
    INCLUDE_PRIVATE)

  nanobind_add_stub(
    fomac_stub
    INSTALL_TIME
    MODULE
    mqt.core.fomac
    OUTPUT
    fomac.pyi
    PYTHON_PATH
    ${CMAKE_INSTALL_PREFIX}/../..
    MARKER_FILE
    py.typed
    INCLUDE_PRIVATE)

  nanobind_add_stub(
    na_fomac_stub
    INSTALL_TIME
    MODULE
    mqt.core.na.fomac
    OUTPUT
    na/fomac.pyi
    PYTHON_PATH
    ${CMAKE_INSTALL_PREFIX}/../..
    MARKER_FILE
    na/py.typed
    INCLUDE_PRIVATE)

  file(GLOB_RECURSE PYI_FILES ${CMAKE_INSTALL_PREFIX}/*.pyi)
  install(
    FILES ${PYI_FILES}
    DESTINATION .
    COMPONENT ${MQT_CORE_TARGET_NAME}_Python)

  file(GLOB_RECURSE PY_TYPED_FILES ${CMAKE_INSTALL_PREFIX}/py.typed)
  install(
    FILES ${PY_TYPED_FILES}
    DESTINATION .
    COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
endif()
