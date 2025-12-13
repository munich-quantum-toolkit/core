# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS FluxToQuartz.td)
mlir_tablegen(FluxToQuartz.h.inc -gen-pass-decls -name FluxToQuartz)
add_public_tablegen_target(FluxToQuartzIncGen)

add_mlir_doc(FluxToQuartz MLIRFluxToQuartz Conversions/ -gen-pass-doc)
