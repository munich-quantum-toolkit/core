# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(devices)
if(BUILD_MQT_CORE_QDMI_DDSIM_DEVICE AND BUILD_MQT_CORE_QDMI_SC_DEVICE)
  add_subdirectory(driver)
endif()
add_subdirectory(registry)
