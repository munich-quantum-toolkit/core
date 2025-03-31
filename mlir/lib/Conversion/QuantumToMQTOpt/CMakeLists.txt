# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB CONVERSION_SOURCES *.cpp)

set(LIBRARIES ${conversion_libs} MLIRIR MLIRPass MQT::CoreIR ${MLIRQuantum_LIB})

add_mlir_library(CatalystQuantumToMQTOpt ${CONVERSION_SOURCES} LINK_LIBS ${LIBRARIES} DEPENDS
                 QuantumToMQTOptIncGen)
