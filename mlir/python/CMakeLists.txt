# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Make sure Python (with the development module component) is available for the nanobind extension.
# nanobind requires the Python::Module target.
if(NOT TARGET Python::Module)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
endif()

# Locate nanobind (provided via the active Python environment).
if(NOT TARGET nanobind-static)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
endif()

nanobind_add_module(MQTCoreMLIRPythonModule NB_SUPPRESS_WARNINGS MQTCoreModule.cpp)
set_target_properties(MQTCoreMLIRPythonModule PROPERTIES OUTPUT_NAME _mqtCore)
target_compile_features(MQTCoreMLIRPythonModule PRIVATE cxx_std_20)
target_link_libraries(MQTCoreMLIRPythonModule PRIVATE MQTCoreCAPI MLIRIR)
mqt_mlir_apply_target_options(MQTCoreMLIRPythonModule)

# Place the extension next to the Python sources so it can be imported as mqt.core.mlir._mqtCore
# from the build tree.
set(MQT_CORE_MLIR_PYTHON_DIR ${CMAKE_BINARY_DIR}/python/mqt/core/mlir)
set_target_properties(MQTCoreMLIRPythonModule PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                         ${MQT_CORE_MLIR_PYTHON_DIR})

# Copy the pure-Python package source next to the compiled extension so the package can be imported
# from the build tree .
add_custom_command(
  TARGET MQTCoreMLIRPythonModule
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different ${PROJECT_SOURCE_DIR}/python/mqt/core/mlir/__init__.py
    ${MQT_CORE_MLIR_PYTHON_DIR}/__init__.py
  COMMENT "Copying mqt.core.mlir Python source to the build tree"
  VERBATIM)
