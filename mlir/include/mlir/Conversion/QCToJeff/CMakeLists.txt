# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS QCToJeff.td)
mlir_tablegen(QCToJeff.h.inc -gen-pass-decls -name QCToJeff)
add_public_tablegen_target(QCToJeffIncGen)

add_mlir_doc(QCToJeff MLIRQCToJeff Conversions/ -gen-pass-doc)
