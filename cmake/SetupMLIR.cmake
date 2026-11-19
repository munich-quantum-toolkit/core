# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# set the include directory for the build tree
set(MQT_MLIR_SOURCE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/mlir/include")
set(MQT_MLIR_BUILD_INCLUDE_DIR "${PROJECT_BINARY_DIR}/mlir/include")
set(_mqt_mlir_series "23.1")
if(NOT MQT_CORE_RELEASE_WHEEL)
  set(MQT_MLIR_MIN_VERSION "${_mqt_mlir_series}" CACHE STRING "Required MLIR release series")
  set(_mqt_mlir_series "${MQT_MLIR_MIN_VERSION}")
endif()
option(MQT_CORE_AUTO_INSTALL_MLIR "Download LLVM/MLIR when no compatible SDK is found" ON)
set(MQT_CORE_MLIR_VERSION "23.1.4" CACHE STRING "LLVM/MLIR version to download")
get_property(_mqt_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _mqt_build_type)
set(_mqt_mlir_assertions_default OFF)
if(NOT _mqt_multi_config AND _mqt_build_type MATCHES "^(DEBUG|RELWITHDEBINFO)$")
  set(_mqt_mlir_assertions_default ON)
endif()
option(MQT_CORE_MLIR_ASSERTIONS "Use assertions in the downloaded LLVM/MLIR SDK"
       ${_mqt_mlir_assertions_default})

# Explicit SDK paths take precedence over package search prefixes and the local .env.
foreach(_mqt_package IN ITEMS LLVM MLIR)
  if(NOT ${_mqt_package}_DIR AND NOT "$ENV{${_mqt_package}_DIR}" STREQUAL "")
    set(${_mqt_package}_DIR "$ENV{${_mqt_package}_DIR}")
  endif()
endforeach()
if(MQT_CORE_RELEASE_WHEEL AND LLVM_DIR AND NOT MLIR_DIR)
  set(MLIR_DIR "${LLVM_DIR}/../mlir")
endif()
if(NOT MLIR_DIR AND EXISTS "${PROJECT_SOURCE_DIR}/.env")
  file(STRINGS "${PROJECT_SOURCE_DIR}/.env" MQT_CORE_DOTENV_LINES)
  foreach(MQT_CORE_DOTENV_LINE IN LISTS MQT_CORE_DOTENV_LINES)
    if(MQT_CORE_DOTENV_LINE MATCHES
       "^[ \t]*(export[ \t]+)?MLIR_DIR[ \t]*=[ \t]*['\"]?([^'\"]+)['\"]?[ \t]*$")
      file(TO_CMAKE_PATH "${CMAKE_MATCH_2}" MLIR_DIR)
      message(STATUS "Using MLIR_DIR from ${PROJECT_SOURCE_DIR}/.env: ${MLIR_DIR}")
      break()
    endif()
  endforeach()
endif()

# Portable SDKs install both package configs together.
if(MLIR_DIR AND NOT LLVM_DIR AND EXISTS "${MLIR_DIR}/../llvm/LLVMConfig.cmake")
  set(LLVM_DIR "${MLIR_DIR}/../llvm")
endif()

# Imported targets cannot be replaced by another SDK during configuration.
if(TARGET LLVMSupport OR TARGET MLIRSupport)
  if(NOT LLVM_PACKAGE_VERSION
     OR NOT "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}" VERSION_EQUAL _mqt_mlir_series)
    message(FATAL_ERROR "The embedding project must provide a compatible LLVM/MLIR SDK.")
  endif()
  find_package(MLIR ${LLVM_PACKAGE_VERSION} EXACT CONFIG REQUIRED)
elseif(NOT MQT_CORE_RELEASE_WHEEL)
  find_package(MLIR ${_mqt_mlir_series} CONFIG QUIET)
endif()
if(MQT_CORE_RELEASE_WHEEL OR NOT MLIR_FOUND)
  if(NOT MQT_CORE_MLIR_VERSION MATCHES "^([0-9]+\\.[0-9]+)\\.[0-9]+$"
     OR NOT CMAKE_MATCH_1 VERSION_EQUAL _mqt_mlir_series)
    message(FATAL_ERROR "MQT_CORE_MLIR_VERSION must be a ${_mqt_mlir_series}.x release.")
  endif()
endif()
if(MQT_CORE_RELEASE_WHEEL AND NOT MLIR_FOUND AND MLIR_DIR)
  # A release wheel can reuse an explicitly supplied portable SDK, including offline.
  find_package(MLIR ${MQT_CORE_MLIR_VERSION} EXACT CONFIG REQUIRED NO_DEFAULT_PATH)
endif()
if(NOT MLIR_FOUND)
  if(MQT_CORE_MLIR_ASSERTIONS AND NOT MQT_CORE_RELEASE_WHEEL)
    set(_mqt_mlir_assertions ON)
  else()
    set(_mqt_mlir_assertions OFF)
  endif()
  set(_mqt_mlir_prefix
      "${PROJECT_BINARY_DIR}/_deps/llvm-${MQT_CORE_MLIR_VERSION}-${_mqt_mlir_assertions}")

  # shortcut: setup-mlir selects the host SDK; target selection needs upstream cross-build support.
  if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "Automatic LLVM/MLIR setup supports native builds only. Set MLIR_DIR.")
  endif()

  if(NOT EXISTS "${_mqt_mlir_prefix}/.complete")
    if(NOT MQT_CORE_AUTO_INSTALL_MLIR OR FETCHCONTENT_FULLY_DISCONNECTED)
      message(
        FATAL_ERROR "No compatible LLVM/MLIR SDK found and downloading is disabled. Set MLIR_DIR.")
    endif()
    include(FetchContent)
    # Keep the installer and its version manifest at the same revision.
    fetchcontent_declare(
      setup_mlir
      URL
        https://github.com/munich-quantum-software/setup-mlir/archive/246aa0c70414109aa55fa19a5370739519833512.tar.gz
      URL_HASH SHA256=d780828010b060c85397cae120ffd7803066e45d83b212a5be51baa9d7a71fe6)
    fetchcontent_makeavailable(setup_mlir)
    if(WIN32)
      set(_mqt_mlir_command
          pwsh
          -NoProfile
          -ExecutionPolicy
          Bypass
          -File
          "${setup_mlir_SOURCE_DIR}/installation/setup-mlir.ps1"
          -llvm_version
          "${MQT_CORE_MLIR_VERSION}"
          -install_prefix
          "${_mqt_mlir_prefix}")
      if(NOT _mqt_mlir_assertions)
        list(APPEND _mqt_mlir_command -no_assertions)
      endif()
    else()
      set(_mqt_mlir_command
          bash
          "${setup_mlir_SOURCE_DIR}/installation/setup-mlir.sh"
          -v
          "${MQT_CORE_MLIR_VERSION}"
          -p
          "${_mqt_mlir_prefix}"
          -a
          "${_mqt_mlir_assertions}")
    endif()
    message(
      STATUS
        "Installing LLVM/MLIR ${MQT_CORE_MLIR_VERSION} (assertions ${_mqt_mlir_assertions}) in ${_mqt_mlir_prefix}"
    )
    execute_process(COMMAND ${_mqt_mlir_command} COMMAND_ERROR_IS_FATAL ANY)
    file(WRITE "${_mqt_mlir_prefix}/.complete" "")
  endif()

  # Do not cache these paths: changing the requested SDK must select a new prefix.
  set(LLVM_DIR "${_mqt_mlir_prefix}/lib/cmake/llvm")
  set(MLIR_DIR "${_mqt_mlir_prefix}/lib/cmake/mlir")
  find_package(MLIR ${MQT_CORE_MLIR_VERSION} EXACT CONFIG REQUIRED)
endif()
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM/MLIR ${MLIR_VERSION}, assertions ${LLVM_ENABLE_ASSERTIONS}")

# LLVM component targets do not propagate headers to native consumers.
target_include_directories(LLVMSupport SYSTEM INTERFACE "$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(MQTLLVMABI)
mqt_get_llvm_abi_checks(MQT_CORE_LLVM_ABI_CHECKS)
if(MQT_CORE_RELEASE_WHEEL)
  if(NOT MLIR_VERSION VERSION_EQUAL MQT_CORE_MLIR_VERSION
     OR NOT LLVM_PACKAGE_VERSION VERSION_EQUAL MQT_CORE_MLIR_VERSION
     OR LLVM_ENABLE_ASSERTIONS OR MQT_CORE_LLVM_ABI_CHECKS)
    message(
      FATAL_ERROR
        "Release wheels require LLVM/MLIR ${MQT_CORE_MLIR_VERSION} with assertions and ABI-breaking checks disabled. "
        "Found MLIR ${MLIR_VERSION} and LLVM ${LLVM_PACKAGE_VERSION} with assertions ${LLVM_ENABLE_ASSERTIONS} and ABI-breaking checks ${MQT_CORE_LLVM_ABI_CHECKS}."
    )
  endif()
  if(NOT MLIR_INSTALL_PREFIX STREQUAL LLVM_INSTALL_PREFIX)
    message(
      FATAL_ERROR
        "Release wheels require LLVM and MLIR from the same SDK. Check MLIR_DIR and LLVM_DIR.")
  endif()
endif()

# Keep direct LLVM consumers compatible with an RTTI-free LLVM installation. Ordinary MLIR targets
# receive this policy through llvm_update_compile_flags.
function(mqt_llvm_target_disable_rtti target_name)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "Cannot configure RTTI for missing target ${target_name}.")
  endif()
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(
      ${target_name}
      PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${LLVM_CXXFLAGS_RTTI_DISABLE}>)
  endif()
endfunction()

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${MQT_MLIR_SOURCE_INCLUDE_DIR})
include_directories(${MQT_MLIR_BUILD_INCLUDE_DIR})
separate_arguments(MQT_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${MQT_LLVM_DEFINITIONS})

# Generate MLIR artifacts, including documentation, in the build tree.
set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
