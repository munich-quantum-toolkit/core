# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_library(MQTCompilerCatalystAdapter STATIC QDMIDevice.cpp)
set_target_properties(MQTCompilerCatalystAdapter PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  MQTCompilerCatalystAdapter
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(
  MQTCompilerCatalystAdapter
  SYSTEM
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/vendor/catalyst/0.15.0/include>")
target_link_libraries(
  MQTCompilerCatalystAdapter
  PUBLIC MQTCompilerQDMIAdapter MQT::CoreQDMI
  PRIVATE MQT::ProjectOptions MLIRQCProgramBuilder MLIRQCTranslationSupport)

if(MQT_CORE_INSTALL)
  install(
    FILES "${PROJECT_SOURCE_DIR}/vendor/catalyst/0.15.0/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/mqt-core/catalyst"
    COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
endif()
