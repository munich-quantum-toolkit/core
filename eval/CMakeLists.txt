# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-core-dd-eval eval_dd_package.cpp)
target_link_libraries(
  mqt-core-dd-eval PRIVATE MQT::CoreDD MQT::CoreAlgorithms MQT::CoreCircuitOptimizer
                           MQT::ProjectOptions MQT::ProjectWarnings)
