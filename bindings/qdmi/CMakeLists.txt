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
    .
    LINK_LIBS
    MQT::CoreFoMaC)

  # Set C++ standard
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)

  # install the Python stub files in editable mode for better IDE support
  if(SKBUILD_STATE STREQUAL "editable")
    file(GLOB_RECURSE FOMAC_PYI_FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/qdmi/fomac/*.pyi)
    install(
      FILES ${FOMAC_PYI_FILES}
      DESTINATION ./qdmi
      COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
  endif()
endif()
