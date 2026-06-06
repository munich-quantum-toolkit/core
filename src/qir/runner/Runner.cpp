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

#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/InitLLVM.h>

#include <span>
#include <string>

#define DEBUG_TYPE "mqt-core-qir-runner"

static llvm::codegen::RegisterCodeGenFlags CGF;

static llvm::cl::opt<std::string> InputFile(llvm::cl::desc("<input bitcode>"),
                                            llvm::cl::Positional,
                                            llvm::cl::init("-"));

static llvm::cl::list<std::string>
    InputArgv(llvm::cl::ConsumeAfter, llvm::cl::desc("<program arguments>..."));

static llvm::ExitOnError ExitOnError;

auto main(int argc, char* argv[]) -> int {
  const llvm::InitLLVM session(argc, argv);
  if (const std::span args(argv, argc); args.size() > 1) {
    ExitOnError.setBanner(std::string(args[0]) + ": ");
  }
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "qir interpreter & dynamic compiler\n");

  try {
    auto jitSession = qir::jit::Session(llvm::StringRef(InputFile));
    return jitSession.run(InputArgv, InputFile);
  } catch (const std::exception& e) {
    ExitOnError(llvm::createStringError(e.what()));
  }
}
