# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS MQTQMAP.td)
mlir_tablegen(MQTQMAP.h.inc -gen-pass-decls -name MQTQMAP)
add_public_tablegen_target(MQTQMAPIncGen)

add_mlir_doc(MQTQMAP MQTQMAP ./ -gen-pass-doc)
