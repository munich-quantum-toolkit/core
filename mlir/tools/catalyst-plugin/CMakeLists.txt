# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Detect the installed pennylane-catalyst package.
execute_process(
        COMMAND "${Python_EXECUTABLE}" -m catalyst --include_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE CATALYST_INCLUDE_DIRS)
# Check if the pennylane-catalyst package is installed.
if(NOT CATALYST_INCLUDE_DIRS)
    message(FATAL_ERROR "The catalyst package was not found. Please install it using `pip install pennylane-catalyst`.")
endif()

add_compile_options(-fexceptions)

add_llvm_library(
  mqt-catalyst-plugin
  MODULE
  BUILDTREE_ONLY
  mqt-plugin.cpp
  DEPENDS
  MLIRMQTOpt
  MLIRMQTOptTransforms
  PLUGIN_TOOL
  LINK_LIBS
  MLIRMQTOpt
  MLIRMQTOptTransforms)
