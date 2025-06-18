# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS MQTOptToMQTDyn.td)
mlir_tablegen(MQTOptToMQTDyn.h.inc -gen-pass-decls -name MQTOptToMQTDyn)
add_public_tablegen_target(MQTOptToMQTDynIncGen)

add_mlir_doc(MQTOptToMQTDyn MQTOptToMQTDyn Conversions/ -gen-pass-doc)
