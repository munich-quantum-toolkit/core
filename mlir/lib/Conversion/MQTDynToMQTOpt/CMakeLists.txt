# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

get_property(dialect_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBRARIES ${dialect_libs} MQT::CoreIR)
add_compile_options(-fexceptions)

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_library(MQTDynToMQTOpt ${CONVERSION_SOURCES} LINK_LIBS ${LIBRARIES} DEPENDS
                 MQTDynToMQTOptIncGen)
