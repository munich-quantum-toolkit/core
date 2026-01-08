# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS QCToQIR.td)
mlir_tablegen(QCToQIR.h.inc -gen-pass-decls -name QCToQIR)
add_public_tablegen_target(QCToQIRIncGen)

add_mlir_doc(QCToQIR MLIRQCToQIR Conversions/ -gen-pass-doc)
