# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

function(add_cudaq_dialect_library name)
  add_mlir_library(${ARGV} DISABLE_INSTALL)
  install(TARGETS ${name} COMPONENT ${name})
endfunction()

add_subdirectory(cudaq)
