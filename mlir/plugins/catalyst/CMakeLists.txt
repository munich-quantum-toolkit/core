# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Check if the pennylane-catalyst package is installed in the python environment.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m catalyst --version
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE CATALYST_VERSION)
if(CATALYST_VERSION)
  message(STATUS "Found pennylane-catalyst ${CATALYST_VERSION}")
  # Check if the version is compatible.
  if(CATALYST_VERSION VERSION_LESS 0.12.0-dev22)
    message(
      FATAL_ERROR
        "pennylane-catalyst version ${CATALYST_VERSION} is not compatible. Please install version 0.12.0-dev22 or higher."
    )
  endif()
  # Detect the installed catalyst include files.
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m catalyst --include_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE CATALYST_INCLUDE_DIRS)
else()
  # todo: Add option to get catalyst via fetch_content. Note that catlyst should not be built
  # completely to avoid building entire llvm. Note further that some the headers are only generated
  # during build time.
  message(
    FATAL_ERROR
      "The pennylane-catalyst package was not found. Please install it using, e.g., `pip install pennylane-catalyst`."
  )
endif()

if(NOT CATALYST_INCLUDE_DIRS)
  message(
    FATAL_ERROR
      "The include directory of the pennylane-catalyst package could not be retrieved. Please ensure that the package is installed correctly."
  )
endif()

add_compile_options(-fexceptions)

# todo: Why is it add_llvm_library and not add_mlir_library?
add_llvm_library(
  mqt-catalyst-plugin
  MODULE
  mqt-plugin.cpp
  PLUGIN_TOOL
  LINK_LIBS
  MLIRMQTOpt
  MLIRMQTOptTransforms)

# add MQT alias
add_library(MQT::CatalystPlugin ALIAS mqt-catalyst-plugin)
