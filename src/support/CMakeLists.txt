# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mqt_core_library(${MQT_CORE_TARGET_NAME}-json ALIAS_NAME JSON FORCE_STATIC HIDDEN_VISIBILITY)
target_sources(${MQT_CORE_TARGET_NAME}-json PRIVATE JSON.cpp)
target_sources(
  ${MQT_CORE_TARGET_NAME}-json PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                      FILES ${MQT_CORE_INCLUDE_BUILD_DIR}/support/JSON.hpp)
target_link_libraries(${MQT_CORE_TARGET_NAME}-json PUBLIC LLVMSupport)
target_include_directories(${MQT_CORE_TARGET_NAME}-json SYSTEM
                           PUBLIC $<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>)
mqt_llvm_target_disable_rtti(${MQT_CORE_TARGET_NAME}-json)
set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-json
    PARENT_SCOPE)
