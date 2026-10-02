# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS CBitToMemRef.td)
mlir_tablegen(CBitToMemRef.h.inc -gen-pass-decls -name CBitToMemRef)
add_public_tablegen_target(CBitToMemRefIncGen)

add_mlir_doc(CBitToMemRef CBitToMemRef Conversions/ -gen-pass-doc)
