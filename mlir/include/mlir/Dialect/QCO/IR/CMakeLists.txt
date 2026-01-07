# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect(QCOOps qco)
add_mlir_interface(QCOInterfaces LINK_LIBS PUBLIC Eigen3::Eigen)
add_mlir_doc(QCOOps MLIRQCODialect Dialects/ -gen-dialect-doc)
add_mlir_doc(QCOInterfaces MLIRQCOInterfaces Dialects/ -gen-op-interface-docs -dialect=qco)
