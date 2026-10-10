# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

option(ENABLE_CACHE "Enable compiler and linker caches when supported" ON)
if(NOT ENABLE_CACHE)
  return()
endif()

foreach(language IN ITEMS C CXX)
  # An explicitly empty launcher disables automatic selection for that language.
  if(NOT DEFINED CMAKE_${language}_COMPILER_LAUNCHER)
    find_program(MQT_CORE_COMPILER_CACHE NAMES sccache ccache)
    if(MQT_CORE_COMPILER_CACHE)
      set(CMAKE_${language}_COMPILER_LAUNCHER "${MQT_CORE_COMPILER_CACHE}")
    endif()
  endif()
endforeach()
