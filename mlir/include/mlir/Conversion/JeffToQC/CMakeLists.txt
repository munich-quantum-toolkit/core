# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS JeffToQC.td)
mlir_tablegen(JeffToQC.h.inc -gen-pass-decls -name JeffToQC)
add_public_tablegen_target(JeffToQCIncGen)

add_mlir_doc(JeffToQC MLIRJeffToQC Conversions/ -gen-pass-doc)
