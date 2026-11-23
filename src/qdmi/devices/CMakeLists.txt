# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

get_target_property(QDMI_VERSION qdmi::qdmi VERSION)
if(NOT QDMI_VERSION OR QDMI_VERSION VERSION_LESS QDMI_MINIMUM_VERSION)
  message(FATAL_ERROR "QDMI ${QDMI_MINIMUM_VERSION} or newer is required. Found ${QDMI_VERSION}.")
endif()

add_subdirectory(dd)
if(BUILD_MQT_CORE_QDMI_SC_DEVICE)
  add_subdirectory(sc)
endif()

set(MQT_CORE_TARGETS ${MQT_CORE_TARGETS} PARENT_SCOPE)
