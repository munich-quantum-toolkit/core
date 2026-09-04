# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# add various subdirectories containing tests
add_subdirectory(circuit_optimizer)
add_subdirectory(dd)
add_subdirectory(ir)
add_subdirectory(qdmi)
if(BUILD_MQT_CORE_QDMI_DDSIM_DEVICE AND BUILD_MQT_CORE_QDMI_SC_DEVICE)
  add_subdirectory(fomac)
endif()

# copy test circuits to build directory
file(COPY ${PROJECT_SOURCE_DIR}/test/circuits DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
