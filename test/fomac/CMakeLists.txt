# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-fomac-test)

if(TARGET MQT::CoreQDMIFoMaC)
  package_add_test(${TARGET_NAME} MQT::CoreQDMIFoMaC test_fomac.cpp)
endif()
