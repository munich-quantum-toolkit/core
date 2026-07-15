# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_library(
  MQTOpenQASMParser STATIC Generated/qasm3Lexer.cpp Generated/qasm3Parser.cpp
                           Generated/qasm3ParserBaseVisitor.cpp Generated/qasm3ParserVisitor.cpp)

target_include_directories(
  MQTOpenQASMParser SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Generated
                                  ${antlr4_SOURCE_DIR}/runtime/Cpp/runtime/src)
target_link_libraries(MQTOpenQASMParser PUBLIC antlr4_static)
set_target_properties(MQTOpenQASMParser PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_mlir_library(
  MLIROpenQASMTarget
  OpenQASM.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/mlir/Target/OpenQASM
  LINK_LIBS
  PUBLIC
  MLIROQ3Dialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRMathDialect
  MLIRQCDialect
  MLIRSCFDialect
  MLIRSupport
  MQTOpenQASMParser)

mqt_mlir_target_use_project_options(MLIROpenQASMTarget)

target_sources(
  MLIROpenQASMTarget PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_SOURCE_INCLUDE_DIR} FILES
                            ${MQT_MLIR_SOURCE_INCLUDE_DIR}/mlir/Target/OpenQASM/OpenQASM.h)
