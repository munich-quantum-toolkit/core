# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB TRANSFORMS_SOURCES *.cpp)

set(LIBRARIES MLIRMQTOpt)

add_mlir_library(MQTQMAP ${TRANSFORMS_SOURCES} LINK_LIBS ${LIBRARIES} DEPENDS MQTQMAPIncGen)

target_compile_features(MQTQMAP PUBLIC cxx_std_17)
target_compile_options(MQTQMAP PUBLIC -fexceptions)

target_include_directories(MQTQMAP PRIVATE ${mqt-core_SOURCE_DIR}/include)
target_include_directories(MQTQMAP PRIVATE ${mqt-core_SOURCE_DIR}/include/mqt-core)

message(STATUS "[MQTQMAP] Added MQT Core include: ${mqt-core_SOURCE_DIR}/include")
