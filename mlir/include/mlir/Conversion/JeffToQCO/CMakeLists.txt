# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS JeffToQCO.td)
mlir_tablegen(JeffToQCO.h.inc -gen-pass-decls -name JeffToQCO)
add_public_tablegen_target(JeffToQCOIncGen)

add_mlir_doc(JeffToQCO MLIRJeffToQCO Conversions/ -gen-pass-doc)
