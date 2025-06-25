# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# set common compiler options for projects
function(enable_project_options target_name)
  include(CheckCXXCompilerFlag)

  # Option to enable time tracing with clang
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    option(ENABLE_BUILD_WITH_TIME_TRACE
           "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
    if(ENABLE_BUILD_WITH_TIME_TRACE)
      target_compile_options(${target_name} INTERFACE -ftime-trace)
    endif()
  endif()

  if(MSVC)
    target_compile_options(${target_name} INTERFACE /utf-8 /Zm10)
  else()
    # always include debug symbols (avoids common problems with LTO)
    target_compile_options(${target_name} INTERFACE -g)

    # enable coverage collection options
    option(ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" FALSE)
    if(ENABLE_COVERAGE)
      target_compile_options(${target_name} INTERFACE --coverage -fprofile-arcs -ftest-coverage -O0)
      target_link_libraries(${target_name} INTERFACE gcov --coverage)
    endif()

    if(NOT DEPLOY)
      # only include machine-specific optimizations when building for the host machine
      check_cxx_compiler_flag(-mtune=native HAS_MTUNE_NATIVE)
      if(HAS_MTUNE_NATIVE)
        target_compile_options(${target_name} INTERFACE -mtune=native)
      endif()

      check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
      if(HAS_MARCH_NATIVE)
        if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_SYSTEM_PROCESSOR MATCHES
                                                      "(x86)|(x86_64)|(AMD64)|(amd64)")
          target_compile_options(${target_name} INTERFACE -march=native)
        else()
          target_compile_options(${target_name} INTERFACE -mcpu=native)
        endif()
      endif()
    endif()

    # enable some more optimizations in release mode
    target_compile_options(
      ${target_name} INTERFACE $<$<CONFIG:RELEASE>:-fno-math-errno -ffinite-math-only
                               -fno-trapping-math -fno-stack-protector>)

    # enable some more options for better debugging
    target_compile_options(
      ${target_name} INTERFACE $<$<CONFIG:DEBUG>:-fno-omit-frame-pointer
                               -fno-optimize-sibling-calls -fno-inline-functions>)
  endif()

  option(BINDINGS "Configure for building Python bindings")
  if(BINDINGS)
    include(CheckPIESupported)
    check_pie_supported()
    set_target_properties(${target_name} PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON)
  endif()

  # add a compile definition for _LIBCPP_REMOVE_TRANSITIVE_INCLUDES to remove transitive includes
  # from libc++ headers. This is useful to avoid including system headers that are not needed and
  # that may conflict with other headers. This is only supported by libc++.
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    target_compile_definitions(${target_name} INTERFACE _LIBCPP_REMOVE_TRANSITIVE_INCLUDES)
  endif()
endfunction()
