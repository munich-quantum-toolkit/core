# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect(QCOps qc)
add_mlir_interface(QCInterfaces)
add_mlir_doc(QCOps MLIRQCDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(QCInterfaces MLIRQCInterfaces Dialects/ -gen-op-interface-docs -dialect=qc)
