# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-fomac-test)

if(TARGET MQT::CoreFoMaC)
  package_add_test(${TARGET_NAME} MQT::CoreFoMaC test_fomac.cpp test_slurm.cpp)
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE "MQT_CORE_FOMAC_SLURM_TEST_DEVICE=\"$<TARGET_FILE:mqt-core-qdmi-session-device>\"")
  add_dependencies(${TARGET_NAME} mqt-core-qdmi-session-device)
  mqt_copy_qdmi_runtime(${TARGET_NAME})
endif()
