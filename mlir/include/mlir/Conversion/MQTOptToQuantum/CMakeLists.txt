# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS MQTOptToCatalystQuantum.td)
mlir_tablegen(MQTOptToCatalystQuantum.h.inc -gen-pass-decls -name MQTOptToCatalystQuantum)
add_public_tablegen_target(MQTOptToCatalystQuantumIncGen)

add_mlir_doc(MQTOptToCatalystQuantum MQTOptToCatalystQuantum ./ -gen-pass-doc)
