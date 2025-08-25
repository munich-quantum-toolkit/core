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
    add_dependencies(${TARGET_NAME} MQT::CoreDynNaQDMI)
    target_compile_definitions(${TARGET_NAME}
                               PRIVATE DYN_DEV_LIB="$<TARGET_FILE:MQT::CoreDynNaQDMI>")
  endif()
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
endif()

set(TARGET_NAME mqt-core-fomac-test)

if(TARGET MQT::CoreFoMaC)
  package_add_test(${TARGET_NAME} MQT::CoreFoMaC test_fomac.cpp)
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
endif()
