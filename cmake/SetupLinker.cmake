# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
   OR DEFINED CMAKE_LINKER_TYPE
   OR LLVM_USE_LINKER OR LLVM_ENABLE_LTO)
  return()
endif()

# Some callers select their linker in the compiler or linker flags.
set(_mqt_driver_flags "${CMAKE_C_COMPILER_ARG1} ${CMAKE_CXX_COMPILER_ARG1}")
string(TOUPPER "${CMAKE_BUILD_TYPE};${CMAKE_CONFIGURATION_TYPES}" _mqt_configs)
foreach(_mqt_kind IN ITEMS C CXX EXE_LINKER SHARED_LINKER MODULE_LINKER)
  string(APPEND _mqt_driver_flags " ${CMAKE_${_mqt_kind}_FLAGS}")
  foreach(_mqt_config IN LISTS _mqt_configs)
    string(APPEND _mqt_driver_flags " ${CMAKE_${_mqt_kind}_FLAGS_${_mqt_config}}")
  endforeach()
endforeach()
if(_mqt_driver_flags MATCHES "(^|[ 	])(-fuse-ld|--ld-path)(=|[ 	])")
  return()
endif()

set(mqt_require_sdk_mold "${mqt_linux_wheel}")
if(NOT PROJECT_IS_TOP_LEVEL
   OR CMAKE_VERSION VERSION_LESS 3.29
   OR CMAKE_CROSSCOMPILING OR NOT CMAKE_C_COMPILER_ID STREQUAL CMAKE_CXX_COMPILER_ID)
  return()
endif()

set(_mqt_mold "${LLVM_TOOLS_BINARY_DIR}/ld.mold")
if(NOT EXISTS "${_mqt_mold}")
  return()
endif()
# mold 3 protects shared-library exports during safe identical code folding.
execute_process(COMMAND "${_mqt_mold}" --version OUTPUT_VARIABLE _mqt_mold_version ERROR_QUIET)
if(NOT _mqt_mold_version MATCHES "^mold ([0-9.]+)" OR CMAKE_MATCH_1 VERSION_LESS 3)
  return()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(_mqt_linker_flags "--ld-path=\"${_mqt_mold}\"")
  if(ENABLE_IPO)
    set(_mqt_llvmgold "${LLVM_LIBRARY_DIR}/LLVMgold.so")
    if(NOT EXISTS "${_mqt_llvmgold}"
       OR NOT CMAKE_C_COMPILER_VERSION VERSION_EQUAL LLVM_PACKAGE_VERSION
       OR NOT CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL LLVM_PACKAGE_VERSION)
      return()
    endif()
    list(APPEND _mqt_linker_flags -Xlinker -plugin -Xlinker "\"${_mqt_llvmgold}\"")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND EXISTS "${LLVM_INSTALL_PREFIX}/libexec/mold/ld")
  # Use only the SDK's linker search directory. GCC supplies its own LTO plugin.
  set(_mqt_linker_flags "-B\"${LLVM_INSTALL_PREFIX}/libexec/mold/\"")
else()
  return()
endif()

set(CMAKE_LINKER_TYPE mqt_sdk_mold)
set(CMAKE_C_USING_LINKER_mqt_sdk_mold "${_mqt_linker_flags}")
set(CMAKE_CXX_USING_LINKER_mqt_sdk_mold "${_mqt_linker_flags}")
# The IPO probe needs the same linker and plugin as the real build.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CMAKE_C_USING_LINKER_mqt_sdk_mold
     CMAKE_CXX_USING_LINKER_mqt_sdk_mold)
