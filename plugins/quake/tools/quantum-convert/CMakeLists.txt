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
    MQTDynToQuake)

add_mlir_tool(quantum-convert quantum-convert.cpp DEPENDS ${LIBS} SUPPORT_PLUGINS)
target_compile_options(quantum-convert PRIVATE -fexceptions)
target_link_libraries(quantum-convert PUBLIC ${LIBS} ${CUDAQ_MLIR_RUNTIME_LIB})
llvm_update_compile_flags(quantum-convert)
mlir_check_all_link_libraries(quantum-convert)
export_executable_symbols_for_plugins(quantum-convert)
