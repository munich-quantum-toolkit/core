# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS DelayUnit.td)
mlir_tablegen(DelayUnit.h.inc -gen-enum-decls)
mlir_tablegen(DelayUnit.cpp.inc -gen-enum-defs)
add_mlir_dialect_tablegen_target(MLIRMQTDelayUnitIncGen)
