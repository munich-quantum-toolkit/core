# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

option(ENABLE_CACHE "Enable compiler cache if available" ON)
if(NOT ENABLE_CACHE)
  return()
endif()

set(CACHE_OPTION_VALUES "ccache" "sccache")
set(CACHE_OPTION
    "ccache"
    CACHE STRING "Compiler cache to use")
set_property(CACHE CACHE_OPTION PROPERTY STRINGS ${CACHE_OPTION_VALUES})
list(FIND CACHE_OPTION_VALUES ${CACHE_OPTION} CACHE_OPTION_INDEX)
if(CACHE_OPTION_INDEX EQUAL -1)
  message(NOTICE
          "Unknown compiler cache '${CACHE_OPTION}'. Available options are: ${CACHE_OPTION_VALUES}")
endif()

find_program(CACHE_BINARY ${CACHE_OPTION})
if(CACHE_BINARY)
  message(STATUS "Compiler cache '${CACHE_OPTION}' found and enabled")
  set(CMAKE_C_COMPILER_LAUNCHER ${CACHE_BINARY})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CACHE_BINARY})
else()
  message(NOTICE "${CACHE_OPTION} is enabled but was not found. Not using it")
endif()
