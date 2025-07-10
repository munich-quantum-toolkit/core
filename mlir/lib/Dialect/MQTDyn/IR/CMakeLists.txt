# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect_library(MLIRMQTDyn MQTDynOps.cpp DEPENDS MLIRMQTDynOpsIncGen
                         MLIRMQTDynInterfacesIncGen)

# collect header files
file(GLOB_RECURSE IR_HEADERS ${MQT_MLIR_INCLUDE_BUILD_DIR}/mlir/Dialect/MQTDyn/IR/*.h)

# add public headers using file sets
target_sources(MLIRMQTDyn PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_MLIR_INCLUDE_BUILD_DIR} FILES
                                 ${IR_HEADERS})
