# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS QuakeToMQTDyn.td)
mlir_tablegen(QuakeToMQTDyn.h.inc -gen-pass-decls -name QuakeToMQTDyn)
add_public_tablegen_target(QuakeToMQTDynIncGen)

add_mlir_doc(QuakeToMQTDyn QuakeToMQTDyn ./ -gen-pass-doc)
