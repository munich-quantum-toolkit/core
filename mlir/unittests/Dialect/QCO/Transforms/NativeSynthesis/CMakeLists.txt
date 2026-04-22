# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(target_name mqt-core-mlir-unittest-native-synthesis)
add_executable(
  ${target_name}
  native_synthesis_test_helpers.cpp test_native_synthesis_pass_custom_menus.cpp
  test_native_synthesis_pass_fusion.cpp test_native_synthesis_pass_multi_qubit.cpp
  test_native_synthesis_pass_profiles.cpp test_native_synthesis_pass_scoring.cpp)

target_link_libraries(${target_name} PRIVATE MLIRParser GTest::gtest_main MLIRQCProgramBuilder
                                             MLIRQCToQCO MLIRQCOTransforms)

mqt_mlir_configure_unittest_target(${target_name})

gtest_discover_tests(${target_name} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
