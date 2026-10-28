# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB_RECURSE SOURCES **.cpp)
target_sources(${MQT_CORE_TARGET_NAME}-bindings PRIVATE ${SOURCES})
target_link_libraries(${MQT_CORE_TARGET_NAME}-bindings PRIVATE MQT::CoreQDMI)
