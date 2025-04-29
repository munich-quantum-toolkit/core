# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(DIALECT_NAME "MQTOPT")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../Common/IR/StdOps.td.inc
               ${CMAKE_CURRENT_BINARY_DIR}/MQTOptStdOps.td @ONLY)

add_mlir_dialect(MQTOptOps mqtopt)
add_mlir_interface(MQTOptInterfaces)
add_mlir_doc(MQTOptOps MLIRMQTOptDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(MQTOptInterfaces MLIRMQTOptInterfaces Dialects/ -gen-op-interface-docs)
