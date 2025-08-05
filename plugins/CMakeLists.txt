# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(CMAKE_CXX_STANDARD 20)

if(BUILD_MQT_CORE_QUAKE_PLUGIN)
  add_subdirectory(quake)
endif()
