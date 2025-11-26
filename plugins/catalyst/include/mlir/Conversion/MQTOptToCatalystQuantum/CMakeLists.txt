# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS MQTOptToCatalystQuantum.td)
mlir_tablegen(MQTOptToCatalystQuantum.h.inc -gen-pass-decls -name MQTOptToCatalystQuantum)
add_public_tablegen_target(MQTOptToCatalystQuantumIncGen)

add_mlir_doc(MQTOptToCatalystQuantum MQTOptToCatalystQuantum ./ -gen-pass-doc)
