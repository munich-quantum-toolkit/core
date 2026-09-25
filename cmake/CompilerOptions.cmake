# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# set common compiler options for projects
function(enable_project_options target_name)
  include(CheckCXXCompilerFlag)

  # CMake records the linker ID when first enabling the language.
  set(linker_id "${CMAKE_CXX_COMPILER_LINKER_ID}")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.29 AND CMAKE_LINKER_TYPE)
    set(linker_id "${CMAKE_LINKER_TYPE}")
  endif()

  if(APPLE)
    target_link_options(${target_name} INTERFACE "$<$<NOT:$<CONFIG:Debug>>:LINKER:-dead_strip>")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(${target_name} INTERFACE "$<$<NOT:$<CONFIG:Debug>>:LINKER:--gc-sections>")
    if(linker_id STREQUAL "LLD")
      target_link_options(${target_name} INTERFACE "$<$<NOT:$<CONFIG:Debug>>:LINKER:--icf=safe>")
    endif()
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    if(ENABLE_CACHE AND CMAKE_INTERPROCEDURAL_OPTIMIZATION)
      set(lto_cache "${PROJECT_BINARY_DIR}/thinlto-cache-$<CONFIG>")
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND linker_id STREQUAL "LLD")
        target_link_options(${target_name} INTERFACE "LINKER:--thinlto-cache-dir=${lto_cache}"
                            "LINKER:--thinlto-cache-policy=cache_size_bytes=1g")
      elseif(APPLE AND linker_id MATCHES "^(AppleClang|DEFAULT|SYSTEM|APPLE_CLASSIC)$")
        target_link_options(${target_name} INTERFACE "LINKER:-cache_path_lto,${lto_cache}")
      endif()
    endif()

    option(ENABLE_BUILD_WITH_TIME_TRACE
           "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
    if(ENABLE_BUILD_WITH_TIME_TRACE)
      target_compile_options(${target_name} INTERFACE -ftime-trace)
    endif()
  endif()

  if(MSVC)
    target_compile_options(${target_name} INTERFACE /utf-8 /EHsc)
    target_compile_definitions(${target_name} INTERFACE NOMINMAX)
  else()
    option(ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" FALSE)
    if(ENABLE_COVERAGE)
      target_compile_options(${target_name} INTERFACE --coverage -fprofile-arcs -ftest-coverage -O0)
      target_link_libraries(${target_name} INTERFACE --coverage)
    endif()

    if(NOT DEPLOY AND NOT DEFINED ENV{CI})
      # CI caches can reuse object files on runners with different CPUs.
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
    target_compile_options(${target_name} INTERFACE $<$<CONFIG:RELEASE>:-fno-math-errno
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

  # Expose missing direct includes by disabling libc++'s optional transitive includes.
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    target_compile_definitions(${target_name} INTERFACE _LIBCPP_REMOVE_TRANSITIVE_INCLUDES)
  endif()
endfunction()

# Keep compiler exception policy here. MSVC retains its normal STL and unwind ABI.
function(mqt_target_disable_exceptions target_name)
  if(NOT MSVC)
    target_compile_options(${target_name} PRIVATE -fno-exceptions)
  endif()
endfunction()

# Dependency and language boundaries contain exceptions locally.
function(mqt_source_enable_exceptions source)
  if(MSVC)
    set_source_files_properties(${source} PROPERTIES COMPILE_OPTIONS "/EHsc")
  else()
    set_source_files_properties(${source} PROPERTIES COMPILE_OPTIONS "-fexceptions")
  endif()
endfunction()
