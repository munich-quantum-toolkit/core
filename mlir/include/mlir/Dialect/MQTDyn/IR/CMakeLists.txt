# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect(MQTDynOps mqtdyn)
add_mlir_interface(MQTDynInterfaces)
add_mlir_doc(MQTDynOps MLIRMQTDynDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(MQTDynInterfaces MLIRMQTDynInterfaces Dialects/ -gen-op-interface-docs)
