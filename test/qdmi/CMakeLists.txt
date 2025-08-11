# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-qdmi-driver-test)

if(TARGET MQT::CoreQdmiDriver)
  package_add_test(${TARGET_NAME} MQT::CoreQdmiDriver test_driver.cpp)
  add_dependencies(${TARGET_NAME} MQT::CoreDynNaQDMI)
  target_compile_definitions(${TARGET_NAME} PRIVATE DYN_DEV_LIB="$<TARGET_FILE:MQT::CoreDynNaQDMI>")
endif()
