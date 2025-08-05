# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(DIALECT_NAME "MQTMEM")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../Common/IR/StdOps.td.inc
               ${CMAKE_CURRENT_BINARY_DIR}/MQTMemStdOps.td @ONLY)

add_mlir_dialect(MQTMemOps MQTMem)
# add_mlir_interface(MQTMemInterfaces)
add_mlir_doc(MQTMemOps MLIRMQTMemDialect Dialects/ -gen-dialect-doc)
# add_mlir_doc(MQTMemInterfaces MLIRMQTMemInterfaces Dialects/ -gen-op-interface-docs)
