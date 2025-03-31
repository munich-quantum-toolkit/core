# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Navigate up five levels using relative path notation
get_filename_component(CATALYST_BASE_DIR "${MLIR_DIR}/../../../../.." ABSOLUTE)
# Define the MLIR include directory directly
set(MLIRQUANTUM_INCLUDE_DIR "${CATALYST_BASE_DIR}/mlir/include")

# Global settings for compiler flags
add_compile_options(-fexceptions)

# Add include directory for MLIRQuantum headers
include_directories(${MLIRQUANTUM_INCLUDE_DIR})

# Optionally, if using TableGen-generated headers, include build/include (skip if not depending on
# generated .inc files)
include_directories("${CATALYST_BASE_DIR}/build/include")

# Register conversion subdirectories
add_subdirectory(MQTOptToCatalystQuantum)
add_subdirectory(CatalystQuantumToMQTOpt)
