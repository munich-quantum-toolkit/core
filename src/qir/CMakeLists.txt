# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(BUILD_MQT_CORE_QIR_RUNNER OR BUILD_MQT_CORE_QDMI_DDSIM_WITH_QIR)
  add_subdirectory(jit)
endif()

add_subdirectory(runtime)

if(BUILD_MQT_CORE_QIR_RUNNER)
  add_subdirectory(runner)
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
