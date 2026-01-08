# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS QCToQCO.td)
mlir_tablegen(QCToQCO.h.inc -gen-pass-decls -name QCToQCO)
add_public_tablegen_target(QCToQCOIncGen)

add_mlir_doc(QCToQCO MLIRQCToQCO Conversions/ -gen-pass-doc)
