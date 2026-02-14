# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(MQT_MLIR_UNITTESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(mqt_mlir_configure_unittest_target target_name)
  target_include_directories(${target_name} PRIVATE ${MQT_MLIR_UNITTESTS_DIR}
                                                    ${CMAKE_CURRENT_SOURCE_DIR})
  mqt_mlir_target_use_project_options(${target_name})
  set_target_properties(${target_name} PROPERTIES UNITY_BUILD ON)
endfunction()

add_subdirectory(programs)
add_subdirectory(Compiler)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
