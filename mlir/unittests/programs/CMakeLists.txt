# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_library(MLIRQCPrograms qc_programs.cpp)
target_include_directories(MLIRQCPrograms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  MLIRQCPrograms
  PUBLIC MLIRQCProgramBuilder
  PRIVATE MQT::ProjectOptions)
target_sources(MLIRQCPrograms PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} FILES
                                     qc_programs.h)

add_library(MLIRQCOPrograms qco_programs.cpp)
target_include_directories(MLIRQCOPrograms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  MLIRQCOPrograms
  PUBLIC MLIRQCOProgramBuilder
  PRIVATE MQT::ProjectOptions)
target_sources(MLIRQCOPrograms PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} FILES
                                      qco_programs.h)
