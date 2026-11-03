# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_library(MQTPennyLaneRecorder STATIC CircuitRecorder.cpp)
set_target_properties(MQTPennyLaneRecorder PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  MQTPennyLaneRecorder
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_link_libraries(
  MQTPennyLaneRecorder
  PUBLIC MQTCompilerPipeline
  PRIVATE MQT::ProjectOptions MLIRQCProgramBuilder MLIRQCTranslationSupport)
