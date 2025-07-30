# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB CONVERSION_SOURCES *.cpp)

set(LIBRARIES MLIRMQTOpt)

add_mlir_library(CatalystQuantumToMQTOpt ${CONVERSION_SOURCES} LINK_LIBS ${LIBRARIES} DEPENDS
                 CatalystQuantumToMQTOptIncGen)

target_compile_features(CatalystQuantumToMQTOpt PUBLIC cxx_std_20)
target_compile_options(CatalystQuantumToMQTOpt PUBLIC -fexceptions)
