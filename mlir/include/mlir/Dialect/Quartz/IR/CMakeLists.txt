# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(DIALECT_NAME "Quartz")
set(DIALECT_NAME_UPPER "FLUX")

add_mlir_dialect(QuartzOps quartz)
add_mlir_interface(QuartzInterfaces)
add_mlir_doc(QuartzOps MLIRQuartzDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(QuartzInterfaces MLIRQuartzInterfaces Dialects/ -gen-op-interface-docs -dialect=quartz)
