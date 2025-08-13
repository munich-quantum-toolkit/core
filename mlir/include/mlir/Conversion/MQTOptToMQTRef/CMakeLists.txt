# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS MQTOptToMQTRef.td)
mlir_tablegen(MQTOptToMQTRef.h.inc -gen-pass-decls -name MQTOptToMQTRef)
add_public_tablegen_target(MQTOptToMQTRefIncGen)

add_mlir_doc(MQTOptToMQTRef MQTOptToMQTRef Conversions/ -gen-pass-doc)
