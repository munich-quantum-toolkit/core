# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect(QCOps qc)
add_mlir_interface(QCInterfaces)

add_mlir_doc(QCOps QCDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(QCInterfaces QCInterfaces Dialects/ -gen-op-interface-docs -dialect=qc)
