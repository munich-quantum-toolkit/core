/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/QDMI.hpp"

#include "NativeCommands.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <string_view>

static int printInstalledPath(const char* argv0, llvm::StringRef relativePath) {
  llvm::SmallString<256> path(llvm::sys::fs::getMainExecutable(
      argv0, reinterpret_cast<void*>(&printInstalledPath)));
  llvm::sys::path::remove_filename(path);
  llvm::sys::path::append(path, relativePath);
  llvm::sys::path::remove_dots(path, true);
  if (!llvm::sys::fs::is_directory(path)) {
    llvm::errs() << "Installation directory not found: " << path << '\n';
    return 1;
  }
  llvm::outs() << path << '\n';
  return 0;
}

int main(int argc, char** argv) {
  // On Windows, InitLLVM reconstructs argc/argv from the process command line.
  const llvm::InitLLVM init(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "compile") {
    return mqtCoreRunCompiler(argc - 1, argv + 1);
  }
  if (argc > 1 && (std::string_view(argv[1]) == "bench" ||
                   std::string_view(argv[1]) == "simulate" ||
                   std::string_view(argv[1]) == "run")) {
    llvm::cl::list<std::string> manifests(
        "qdmi-manifest", llvm::cl::Hidden,
        llvm::cl::desc("Register a QDMI provider manifest"),
        llvm::cl::callback([](const std::string& path) {
          llvm::ExitOnError("mqt-core: ")(
              qdmi::builtin_driver::addManifest(path));
        }));
    if (std::string_view(argv[1]) == "bench") {
      return mqtCoreRunBench(argc - 1, argv + 1);
    }
    if (std::string_view(argv[1]) == "simulate" && argc == 4 &&
        std::string_view(argv[2]) == "--worker-socket") {
      return mqtCoreRunWorker(argv[3]);
    }
    return mqtCoreRunProgram(argc - 1, argv + 1);
  }
  if (argc == 2 && std::string_view(argv[1]) == "--version") {
    llvm::outs() << MQT_CORE_VERSION_STRING << '\n';
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--include-dir") {
    return printInstalledPath(argv[0], MQT_CORE_INCLUDE_RELATIVE);
  }
  if (argc == 2 && std::string_view(argv[1]) == "--cmake-dir") {
    return printInstalledPath(argv[0], MQT_CORE_CMAKE_RELATIVE);
  }
  if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
    llvm::outs()
        << "Usage: mqt-core <command> [options]\n\n"
           "Commands:\n"
           "  compile        Compile quantum programs\n"
           "  bench          List, generate, and evaluate benchmarks\n"
           "  simulate       Simulate quantum programs with DDSIM\n"
           "  run            Compile and run programs on a QDMI device\n\n"
           "Options:\n"
           "  --version      Print the MQT Core version\n"
           "  --include-dir  Print the installed native include directory\n"
           "  --cmake-dir    Print the installed CMake package directory\n\n"
           "Use mqt-core <command> --help for command "
           "options.\n";
    return 0;
  }
  return mqtCoreRunCompiler(argc, argv);
}
