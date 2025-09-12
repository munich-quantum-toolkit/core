# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(DIALECT_NAME "MQTRef")
set(DIALECT_NAME_UPPER "MQTREF")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../Common/IR/StdOps.td.inc
               ${CMAKE_CURRENT_BINARY_DIR}/MQTRefStdOps.td @ONLY)

add_mlir_dialect(MQTRefOps mqtref)
add_mlir_interface(MQTRefInterfaces)
add_mlir_doc(MQTRefOps MLIRMQTRefDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(MQTRefInterfaces MLIRMQTRefInterfaces Dialects/ -gen-op-interface-docs -dialect=mqtref)
