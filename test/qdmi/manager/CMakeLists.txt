# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-qdmi-manager-test)

if(TARGET MQT::CoreQDMI)
  package_add_test(${TARGET_NAME} MQT::CoreQDMI test_device_manager.cpp)
  target_compile_definitions(${TARGET_NAME}
                             PRIVATE SC_DEVICE_LIBRARY="$<TARGET_FILE:MQT::CoreQDMIScDevice>")
endif()
