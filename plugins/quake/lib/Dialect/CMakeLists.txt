# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

function(add_cudaq_dialect_library name)
  add_mlir_library(${ARGV} DISABLE_INSTALL)
  install(TARGETS ${name} COMPONENT ${name})
endfunction()

add_subdirectory(CC)
add_subdirectory(Quake)
