# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS CatalystQuantumToMQTOpt.td)
mlir_tablegen(CatalystQuantumToMQTOpt.h.inc -gen-pass-decls -name CatalystQuantumToMQTOpt)
add_public_tablegen_target(CatalystQuantumToMQTOptIncGen)

add_mlir_doc(CatalystQuantumToMQTOpt CatalystQuantumToMQTOpt ./ -gen-pass-doc)
