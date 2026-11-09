# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_library(MQTQDMIRuntime STATIC QDMIReadout.cpp)
set_target_properties(MQTQDMIRuntime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(MQTQDMIRuntime PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_link_libraries(
  MQTQDMIRuntime
  PUBLIC MQTCompilerPipeline MQT::CoreQDMI
  PRIVATE MQT::ProjectOptions)

add_subdirectory(PennyLane)
add_subdirectory(Catalyst)
