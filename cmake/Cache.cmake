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

# Check for available cache programs, preferring sccache
find_program(SCCACHE_BINARY sccache)
find_program(CCACHE_BINARY ccache)

if(SCCACHE_BINARY)
  set(CACHE_OPTION "sccache")
  set(CACHE_BINARY ${SCCACHE_BINARY})
  message(STATUS "Compiler cache 'sccache' found and enabled")
elseif(CCACHE_BINARY)
  set(CACHE_OPTION "ccache")
  set(CACHE_BINARY ${CCACHE_BINARY})
  message(STATUS "Compiler cache 'ccache' found and enabled")
else()
  set(CACHE_OPTION_VALUES "ccache" "sccache")
  message(NOTICE "No compiler cache found. Checked for: ${CACHE_OPTION_VALUES}")
  return()
endif()

set(CMAKE_C_COMPILER_LAUNCHER ${CACHE_BINARY})
set(CMAKE_CXX_COMPILER_LAUNCHER ${CACHE_BINARY})

if(WIN32 AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)
endif()
