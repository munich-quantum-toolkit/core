# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS QCOToJeff.td)
mlir_tablegen(QCOToJeff.h.inc -gen-pass-decls -name QCOToJeff)
add_public_tablegen_target(QCOToJeffIncGen)

add_mlir_doc(QCOToJeff MLIRQCOToJeff Conversions/ -gen-pass-doc)
