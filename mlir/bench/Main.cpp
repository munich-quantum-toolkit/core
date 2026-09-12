/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "MQTCoreBench.h"

#include <llvm/Support/raw_ostream.h>

#include <exception>

int main(int argc, char** argv) {
  try {
    return runMQTCoreBench(argc, argv);
  } catch (const std::exception& exception) {
    llvm::errs() << exception.what() << '\n';
    return 1;
  }
}
