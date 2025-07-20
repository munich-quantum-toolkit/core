# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET dd)
  # collect source files
  file(GLOB_RECURSE DD_SOURCES **.cpp)

  # declare the Python module
  add_mqt_python_binding(
    CORE
    ${MQT_CORE_TARGET_NAME}-dd-bindings
    ${DD_SOURCES}
    MODULE_NAME
    dd
    INSTALL_DIR
    .
    LINK_LIBS
    MQT::CoreDD)

  # install the Python stub file in editable mode for better IDE support
  if(SKBUILD_STATE STREQUAL "editable")
    install(
      FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/dd.pyi
      DESTINATION .
      COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
  endif()
endif()
