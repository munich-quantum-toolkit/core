# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME "${MQT_CORE_TARGET_NAME}-qdmi-bindings")

if(NOT TARGET ${TARGET_NAME})
  file(GLOB_RECURSE SOURCES **.cpp)

  add_mqt_python_binding(
    CORE
    ${TARGET_NAME}
    ${SOURCES}
    MODULE_NAME
    qdmi
    INSTALL_DIR
    qdmi
    LINK_LIBS
    MQT::CoreQDMI)

  # A native package keeps the public module name while allowing Python submodules.
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME __init__)

endif()
