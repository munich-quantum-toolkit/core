# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Disable versioned soname — avoids duplication in Python wheels.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

set(MQT_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mqt/core/mlir")
set(MQT_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")

# ##################################################################################################
# Python sources
# ##################################################################################################

declare_mlir_python_sources(
  MQTCoreMLIRPythonSources
  ROOT_DIR "${MQT_MLIR_PYTHON_ROOT_DIR}"
  SOURCES __init__.py _pipeline.py)

declare_mlir_python_sources(MQTCoreMLIRPythonExtensions)

# ##################################################################################################
# Nanobind extension module
# ##################################################################################################

declare_mlir_python_extension(
  MQTCoreMLIRPythonExtensions.Main
  MODULE_NAME
  _mqtCoreMlir
  ADD_TO_PARENT
  MQTCoreMLIRPythonExtensions
  SOURCES
  MQTCoreMLIRModule.cpp
  EMBED_CAPI_LINK_LIBS
  MQTMLIRCoreDialectsCAPI
  MLIRQCTranslation
  MLIRSupportMQT
  PRIVATE_LINK_LIBS
  LLVMSupport
  MQT::CoreIR
  MQTCompilerPipeline)

# ##################################################################################################
# Aggregate CAPI and install
# ##################################################################################################

set(_source_components MQTCoreMLIRPythonSources MQTCoreMLIRPythonExtensions)

add_mlir_python_common_capi_library(
  MQTCoreMLIRAggregateCAPI
  INSTALL_COMPONENT MQTCoreMLIRPythonModules
  INSTALL_DESTINATION python_packages/mqt/core/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${MQT_MLIR_PYTHON_PACKAGES_DIR}/mqt/core/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../../"
  DECLARED_SOURCES ${_source_components})

add_mlir_python_modules(
  MQTCoreMLIRPythonModules
  ROOT_PREFIX "${MQT_MLIR_PYTHON_PACKAGES_DIR}/mqt/core/mlir"
  INSTALL_PREFIX "python_packages/mqt/core/mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS MQTCoreMLIRAggregateCAPI)
