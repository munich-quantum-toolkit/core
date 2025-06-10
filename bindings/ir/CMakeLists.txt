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
  add_mqt_core_binding(
    ${MQT_CORE_TARGET_NAME}-ir-bindings
    ${IR_SOURCES}
    MODULE_NAME
    ir
    INSTALL_DIR
    .
    LINK_LIBS
    MQT::CoreIR
    MQT::CoreQASM)

  # install the Python stub files in editable mode for better IDE support
  if(SKBUILD_STATE STREQUAL "editable")
    file(GLOB_RECURSE IR_PYI_FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/ir/*.pyi)
    install(
      FILES ${IR_PYI_FILES}
      DESTINATION ./ir
      COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
  endif()
endif()
