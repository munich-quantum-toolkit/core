# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# LLVM exports the ABI policy and the assertion setting separately.
function(mqt_get_llvm_abi_checks result)
  string(TOUPPER "${LLVM_ABI_BREAKING_CHECKS}" mode)
  if(mode STREQUAL "FORCE_ON" OR (mode STREQUAL "WITH_ASSERTS" AND LLVM_ENABLE_ASSERTIONS))
    set(${result} ON PARENT_SCOPE)
  else()
    set(${result} OFF PARENT_SCOPE)
  endif()
endfunction()
