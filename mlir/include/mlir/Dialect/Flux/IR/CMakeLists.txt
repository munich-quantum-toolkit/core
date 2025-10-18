# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(DIALECT_NAME "Flux")
set(DIALECT_NAME_UPPER "FLUX")

add_mlir_dialect(FluxOps flux)
add_mlir_interface(FluxInterfaces)
add_mlir_doc(FluxOps MLIRFluxDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(FluxInterfaces MLIRFluxInterfaces Dialects/ -gen-op-interface-docs -dialect=flux)
