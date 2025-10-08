# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-qdmi-driver-test)

if(TARGET MQT::CoreQDMIDriver)
  package_add_test(${TARGET_NAME} MQT::CoreQDMIDriver test_driver.cpp)
  if(NOT WIN32)
    add_dependencies(${TARGET_NAME} MQT::CoreQDMINaDeviceDyn)
    target_compile_definitions(${TARGET_NAME}
                               PRIVATE DYN_DEV_LIB="$<TARGET_FILE:MQT::CoreQDMINaDeviceDyn>")
  endif()
endif()
