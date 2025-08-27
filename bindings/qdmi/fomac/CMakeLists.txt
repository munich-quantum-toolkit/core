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
  add_mqt_python_binding(
    CORE
    ${TARGET_NAME}
    ${SOURCES}
    MODULE_NAME
    fomac
    INSTALL_DIR
    ./qdmi
    LINK_LIBS
    MQT::CoreFoMaC)

  # Set C++ standard
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
endif()
