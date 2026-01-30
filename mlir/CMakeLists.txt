# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 20)
  set(CMAKE_CXX_STANDARD
      20
      CACHE STRING "C++ standard to conform to")
endif()

if(MSVC)
  add_compile_definitions(_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING)
endif()

if(ENABLE_COVERAGE)
  add_compile_options(--coverage -O0)
  add_link_options(--coverage)
endif()

# add main library code
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# add test code
if(BUILD_MQT_CORE_TESTS)
  add_subdirectory(unittests)
endif()
