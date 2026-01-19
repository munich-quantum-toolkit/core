# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-fomac-test)

if(TARGET MQT::CoreFoMaC)
  package_add_test(${TARGET_NAME} MQT::CoreFoMaC test_fomac.cpp)
endif()
