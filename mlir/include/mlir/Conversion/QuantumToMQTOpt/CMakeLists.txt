# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS CatalystQuantumToMQTOpt.td)
mlir_tablegen(CatalystQuantumToMQTOpt.h.inc -gen-pass-decls -name CatalystQuantumToMQTOpt)
add_public_tablegen_target(QuantumToMQTOptIncGen)

add_mlir_doc(CatalystQuantumToMQTOpt CatalystQuantumToMQTOpt ./ -gen-pass-doc)
