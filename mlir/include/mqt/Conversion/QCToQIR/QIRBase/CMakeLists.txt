# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS QCToQIRBase.td)
mlir_tablegen(QCToQIRBase.h.inc -gen-pass-decls -name QCToQIRBase)
add_public_tablegen_target(QCToQIRBaseIncGen)

add_mlir_doc(QCToQIRBase QCToQIRBase Conversions/ -gen-pass-doc)
