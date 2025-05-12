# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(quantum-opt)
if(BUILD_MQT_CORE_CATALYST_PLUGIN)
  add_subdirectory(catalyst-plugin)
endif()
