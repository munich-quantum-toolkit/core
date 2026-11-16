# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-mlir-unittests-catalyst test_qdmi_device.cpp)
target_link_libraries(
  mqt-core-mlir-unittests-catalyst
  PRIVATE MQTCompilerCatalystAdapter GTest::gtest_main)
target_include_directories(
  mqt-core-mlir-unittests-catalyst
  PRIVATE ${PROJECT_SOURCE_DIR}/src/qdmi/driver)
mqt_mlir_configure_unittest_target(mqt-core-mlir-unittests-catalyst REQUIRES_EH)
mqt_copy_qdmi_runtime(mqt-core-mlir-unittests-catalyst MQT::CoreQDMI_DDSIM_Device)
gtest_discover_tests(
  mqt-core-mlir-unittests-catalyst
  PROPERTIES LABELS mqt-mlir-unittests
  DISCOVERY_TIMEOUT 60)
