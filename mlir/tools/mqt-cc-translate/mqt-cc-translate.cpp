/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Translation/Translation.h"

#include <mlir/Support/LLVM.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>

using namespace mlir;

int main(int argc, char** argv) {
  registerQASM3ToQCTranslation();
  return static_cast<int>(
      failed(mlirTranslateMain(argc, argv, "mqt-cc translation tool")));
}
