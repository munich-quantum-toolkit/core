# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# enable support for all kinds of sanitizers
function(enable_sanitizers target_name)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(sanitizers "")

    option(ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" FALSE)
    if(ENABLE_SANITIZER_ADDRESS)
      list(APPEND sanitizers "address")
    endif()

    option(ENABLE_SANITIZER_LEAK "Enable leak sanitizer" FALSE)
    if(ENABLE_SANITIZER_LEAK)
      list(APPEND sanitizers "leak")
    endif()

    option(ENABLE_SANITIZER_UNDEFINED_BEHAVIOR "Enable undefined behavior sanitizer" FALSE)
    if(ENABLE_SANITIZER_UNDEFINED_BEHAVIOR)
      list(APPEND sanitizers "undefined")
    endif()

    option(ENABLE_SANITIZER_THREAD "Enable thread sanitizer" FALSE)
    if(ENABLE_SANITIZER_THREAD)
      if("address" IN_LIST sanitizers OR "leak" IN_LIST sanitizers)
        message(WARNING "Thread sanitizer does not work with Address and Leak sanitizer enabled")
      else()
        list(APPEND sanitizers "thread")
      endif()
    endif()

    option(ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" FALSE)
    if(ENABLE_SANITIZER_MEMORY AND CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
      if("address" IN_LIST sanitizers
         OR "thread" IN_LIST sanitizers
         OR "leak" IN_LIST sanitizers)
        message(
          WARNING "Memory sanitizer does not work with Address, Thread and Leak sanitizer enabled")
      else()
        list(APPEND sanitizers "memory")
      endif()
    endif()

    list(JOIN sanitizers "," list_of_sanitizers)
  endif()

  if(list_of_sanitizers)
    if(NOT "${list_of_sanitizers}" STREQUAL "")
      target_compile_options(${target_name} INTERFACE -fsanitize=${list_of_sanitizers})
      target_link_options(${target_name} INTERFACE -fsanitize=${list_of_sanitizers})
    endif()
  endif()
endfunction()
