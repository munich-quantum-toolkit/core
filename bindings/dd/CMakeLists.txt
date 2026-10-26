# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB_RECURSE DD_SOURCES **.cpp)

target_sources(${MQT_CORE_TARGET_NAME}-bindings PRIVATE ${DD_SOURCES})
target_link_libraries(${MQT_CORE_TARGET_NAME}-bindings PRIVATE MQT::CoreDD)

# install the Python stub file in editable mode for better IDE support
if(MQT_CORE_INSTALL AND SKBUILD_STATE STREQUAL "editable")
  install(
    FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/dd.pyi
    DESTINATION .
    COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
endif()
