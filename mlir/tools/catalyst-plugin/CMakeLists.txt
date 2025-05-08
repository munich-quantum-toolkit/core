# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Check if the pennylane-catalyst package is installed.
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
else()
  message(
    FATAL_ERROR
      "The pennylane-catalyst package was not found. Please install it using, e.g., `pip install pennylane-catalyst`."
  )
endif()

# Detect the installed pennylane-catalyst package.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m catalyst --include_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE CATALYST_INCLUDE_DIRS)
if(NOT CATALYST_INCLUDE_DIRS)
  message(
    FATAL_ERROR
      "The include directory of the pennylane-catalyst package could not be retrieved. Please ensure that the package is installed correctly."
  )
endif()

add_compile_options(-fexceptions)

add_llvm_library(
  mqt-catalyst-plugin
  MODULE
  mqt-plugin.cpp
  DEPENDS
  MLIRMQTOpt
  MLIRMQTOptTransforms
  PLUGIN_TOOL
  LINK_LIBS
  MLIRMQTOpt
  MLIRMQTOptTransforms)

# add MQT alias
add_library(MQT::CatalystPlugin ALIAS mqt-catalyst-plugin)

# set versioning information
set_target_properties(
  mqt-catalyst-plugin
  PROPERTIES VERSION ${PROJECT_VERSION}
             SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
             EXPORT_NAME CatalystPlugin)
