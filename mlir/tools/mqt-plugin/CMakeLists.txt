# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(PLUGIN_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/mqt_plugin_wheel/mqt_plugin")
add_compile_options(-fexceptions)

add_llvm_library(
  MQTPlugin
  MODULE
  BUILDTREE_ONLY
  mqt-plugin.cpp
  DEPENDS
  MLIRMQTOpt
  MLIRMQTOptTransforms
  PLUGIN_TOOL
  catalyst-cli
  LINK_LIBS
  MLIRMQTOpt
  MLIRMQTOptTransforms
  MQTOptToCatalystQuantum
  CatalystQuantumToMQTOpt
  MLIRQuantum)

# Set output directory for the plugin to be shipped with the wheel
set_target_properties(MQTPlugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PLUGIN_OUTPUT_DIR}")
