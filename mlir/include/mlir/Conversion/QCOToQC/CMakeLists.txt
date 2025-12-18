# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS QCOToQC.td)
mlir_tablegen(QCOToQC.h.inc -gen-pass-decls -name QCOToQC)
add_public_tablegen_target(QCOToQCIncGen)

add_mlir_doc(QCOToQC MLIRQCOToQC Conversions/ -gen-pass-doc)
