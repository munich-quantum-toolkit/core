# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_library(MLIRQCPrograms qc_programs.cpp)
target_link_libraries(MLIRQCPrograms PUBLIC MLIRQCProgramBuilder)
target_sources(MLIRQCPrograms PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} FILES
                                     qc_programs.h)

add_library(MLIRQCOPrograms qco_programs.cpp)
target_link_libraries(MLIRQCOPrograms PUBLIC MLIRQCOProgramBuilder)
target_sources(MLIRQCOPrograms PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} FILES
                                      qco_programs.h)

add_library(MLIRQIRPrograms qir_programs.cpp)
target_link_libraries(MLIRQIRPrograms PUBLIC MLIRQIRProgramBuilder)
target_sources(MLIRQIRPrograms PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} FILES
                                      qir_programs.h)

add_library(MLIRQuantumComputationPrograms quantum_computation_programs.cpp)
target_link_libraries(MLIRQuantumComputationPrograms PUBLIC MQT::CoreIR)
target_sources(
  MLIRQuantumComputationPrograms PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
                                        FILES quantum_computation_programs.h)
