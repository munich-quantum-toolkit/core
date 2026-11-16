# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(MQTResultReadout PARTIAL_SOURCES_INTENDED ResultLayout.cpp LINK_LIBS PUBLIC
                 LLVMSupport)
mqt_mlir_target_use_project_options(MQTResultReadout)
target_include_directories(MQTResultReadout PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_sources(
  MQTResultReadout
  PUBLIC
  FILE_SET HEADERS
    BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR}
    FILES ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mqt/Compiler/ResultLayout.h)

add_library(MQTQDMIRuntime STATIC QDMIReadout.cpp)
set_target_properties(MQTQDMIRuntime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(MQTQDMIRuntime PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_link_libraries(
  MQTQDMIRuntime
  PUBLIC MQTResultReadout MQT::CoreQDMI
  PRIVATE MQT::ProjectOptions)

add_library(MQTQDMIExecution STATIC QDMIContext.cpp)
set_target_properties(MQTQDMIExecution PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(MQTQDMIExecution PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_link_libraries(MQTQDMIExecution PUBLIC MQTCompilerQDMIAdapter PRIVATE MQT::ProjectOptions)

add_subdirectory(PennyLane)
add_subdirectory(Catalyst)
