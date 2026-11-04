/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "NativeCommands.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <string_view>

int main(int argc, char** argv) {
  // On Windows, InitLLVM reconstructs argc/argv from the process command line.
  const llvm::InitLLVM init(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "compile") {
    return mqtCoreRunCompiler(argc - 1, argv + 1);
  }
  if (argc > 1 && std::string_view(argv[1]) == "bench") {
    return mqtCoreRunBench(argc - 1, argv + 1);
  }
  if (argc > 1 && std::string_view(argv[1]) == "simulate") {
    return mqtCoreRunSimulator(argc - 1, argv + 1);
  }
  if (argc == 1 || std::string_view(argv[1]) == "--help") {
    llvm::outs() << "Usage: mqt-cc {compile|bench|simulate} [options]\n";
    return 0;
  }
  return mqtCoreRunCompiler(argc, argv);
}
