# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS MQTOptToQuantum.td)
mlir_tablegen(MQTOptToQuantum.h.inc -gen-pass-decls -name MQTOptToQuantum)
add_public_tablegen_target(MQTOptToQuantumIncGen)

add_mlir_doc(MQTOptToQuantum MQTOptToQuantum ./ -gen-pass-doc)
