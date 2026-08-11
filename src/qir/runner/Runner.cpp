/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/jit/Session.hpp"
#include "qir/runtime/Runtime.hpp"

#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/InitLLVM.h>

#include <cstdint>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>

#define DEBUG_TYPE "mqt-core-qir-runner"

static llvm::codegen::RegisterCodeGenFlags CGF;

static llvm::cl::opt<std::string> InputFile(llvm::cl::desc("<input bitcode>"),
                                            llvm::cl::Positional,
                                            llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    EntryPoint("entry-point", llvm::cl::desc("QIR entry point to execute"));

static llvm::cl::opt<uint64_t>
    Shots("shots", llvm::cl::desc("Number of executions"), llvm::cl::init(1));

static llvm::cl::opt<uint64_t>
    Seed("seed", llvm::cl::desc("Seed for deterministic sampling"));

static llvm::ExitOnError ExitOnError;

auto main(int argc, char* argv[]) -> int {
  const llvm::InitLLVM session(argc, argv);
  if (const std::span args(argv, argc); args.size() > 1) {
    ExitOnError.setBanner(std::string(args[0]) + ": ");
  }
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "qir interpreter & dynamic compiler\n");

  try {
    if (Shots == 0) {
      throw std::invalid_argument("--shots must be greater than zero");
    }
    qir::SessionOptions options;
    if (!EntryPoint.empty()) {
      options.entryPoint = EntryPoint;
    }
    if (Seed.getNumOccurrences() != 0) {
      options.seed = Seed;
    }
    auto jitSession = qir::JitSession(llvm::StringRef(InputFile), options);
    auto& runtime = jitSession.runtime();
    runtime.outputProgramHeader();
    int64_t rc = 0;
    for (uint64_t shot = 0; shot < Shots; ++shot) {
      runtime.outputShotStart();
      rc = jitSession.run();
      runtime.outputShotEnd(rc);
      if (rc != 0) {
        break;
      }
    }
    return static_cast<int>(rc);
  } catch (const std::exception& e) {
    ExitOnError(llvm::createStringError(e.what()));
  }
}
