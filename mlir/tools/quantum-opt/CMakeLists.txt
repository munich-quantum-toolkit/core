# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    ${transform_libs}
    ${passes_libs}
    MLIROptLib
    MLIRMQTOptTransforms
    MLIRMQTDynTransforms)

option(ENABLE_CATALYST "Enable Catalyst conversion passes" ON)
if(ENABLE_CATALYST)
  list(APPEND LIBS CatalystQuantumToMQTOpt MQTOptToCatalystQuantum MLIRQuantum)
endif()

add_mlir_tool(quantum-opt quantum-opt.cpp DEPENDS ${LIBS} SUPPORT_PLUGINS)

if(ENABLE_CATALYST)
  # This makes ENABLE_CATALYST available as a macro in the C++ file.
  target_compile_definitions(quantum-opt PRIVATE ENABLE_CATALYST)
endif()

target_compile_options(quantum-opt PRIVATE -fexceptions)
target_link_libraries(quantum-opt PUBLIC ${LIBS})
llvm_update_compile_flags(quantum-opt)
mlir_check_all_link_libraries(quantum-opt)
export_executable_symbols_for_plugins(quantum-opt)
