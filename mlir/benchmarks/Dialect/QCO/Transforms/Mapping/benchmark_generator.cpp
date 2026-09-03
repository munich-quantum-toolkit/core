/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/Target.h"
#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Dialect/QC/Translation/TranslateQCToOpenQASM3.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/Timer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassInstrumentation.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/Timing.h>
#include <mlir/Transforms/Passes.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <tuple>

using namespace mlir;
using namespace mlir::qco;

/// Return a structured program implementing Grover's algorithm using
/// @p numQubits qubits.
static OwningOpRef<ModuleOp> groverAlg(MLIRContext* context,
                                       const int64_t nqubits,
                                       const int64_t niterations,
                                       const std::string& markedBitstring) {
  QCOProgramBuilder builder(context);
  builder.initialize(SmallVector<Type>(nqubits, builder.getI1Type()));

  SmallVector<Value> qubits(nqubits);
  SmallVector<Value> bits(nqubits);

  Value tensor = builder.qtensorAlloc(static_cast<int64_t>(nqubits));
  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }

  // Initialize: apply Hadamard to all qubits to create uniform superposition
  for (int64_t i = 0; i < nqubits; ++i) {
    qubits[i] = builder.h(qubits[i]);
  }

  // Grover iterations using scf.For
  qubits =
      builder.scfFor(1, niterations, 1, qubits, [&](Value, ValueRange args) {
        SmallVector<Value> bodyQubits(args);
        ArrayRef bodyRef(bodyQubits);

        for (size_t i = 0; i < nqubits; ++i) {
          if (markedBitstring[nqubits - 1 - i] == '0') {
            bodyQubits[i] = builder.x(bodyQubits[i]);
          }
        }

        auto mczOut = builder.mcz(bodyRef.drop_back(), bodyRef.back());
        for (size_t i = 0; i < nqubits - 1; ++i) {
          bodyQubits[i] = mczOut.first[i];
        }
        bodyQubits[nqubits - 1] = mczOut.second;

        for (size_t i = 0; i < nqubits; ++i) {
          if (markedBitstring[nqubits - 1 - i] == '0') {
            bodyQubits[i] = builder.x(bodyQubits[i]);
          }
        }

        //
        // Diffusion operator (2|00..0><00.0| - I)
        //

        for_each(bodyQubits, [&](auto& q) { q = builder.h(q); });
        for_each(bodyQubits, [&](auto& q) { q = builder.x(q); });

        mczOut = builder.mcz(bodyRef.drop_back(), bodyRef.back());
        for (size_t i = 0; i < nqubits - 1; ++i) {
          bodyQubits[i] = mczOut.first[i];
        }
        bodyQubits[nqubits - 1] = mczOut.second;

        for_each(bodyQubits, [&](auto& q) { q = builder.x(q); });
        for_each(bodyQubits, [&](auto& q) { q = builder.h(q); });

        return bodyQubits;
      });

  qubits = builder.barrier(qubits);

  // Measure all qubits
  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  // Clean up
  for (int64_t i = 0; i < nqubits; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto mod = builder.finalize(bits);

  PassManager pm(context);
  pm.addPass(createDecomposeMultiControlled());
  pm.addPass(createCanonicalizerPass());
  pm.run(*mod);

  return mod;
}

static OwningOpRef<ModuleOp>
groverDecomposed(MLIRContext* context, const int64_t nqubits,
                 const int64_t niterations,
                 const std::string& markedBitstring) {
  auto mod = groverAlg(context, nqubits, niterations, markedBitstring);
  PassManager pm(context);
  pm.addPass(createDecomposeMultiControlled());
  pm.addPass(createCanonicalizerPass());
  pm.run(*mod);
  return mod;
}

static void writeMLIR(ModuleOp mod, const std::string& filename) {
  std::error_code ec;
  llvm::raw_fd_ostream out(filename, ec);
  if (ec) {
    llvm::errs() << "Error opening file: " << filename << "\n";
    exit(1);
  }
  mod->print(out);
  out.close();
}

static void writeQASM(ModuleOp mod, const std::string& filename) {
  PassManager pm(mod->getContext());
  pm.addPass(createQCOToQC());
  pm.run(mod);

  std::error_code ec;
  llvm::raw_fd_ostream out(filename, ec);
  if (ec) {
    llvm::errs() << "Error opening file: " << filename << "\n";
    exit(1);
  }

  qc::translateQCToOpenQASM3(mod, out);
  out.close();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: " << argv[0] << " <output_directory>\n";
    return 1;
  }

  std::string outputDir = argv[1];
  std::filesystem::create_directories(outputDir);
  std::filesystem::create_directories(outputDir + "/mlir");
  std::filesystem::create_directories(outputDir + "/qasm");

  MLIRContext context;
  DialectRegistry registry;
  registry.insert<QCODialect, qtensor::QTensorDialect, scf::SCFDialect,
                  arith::ArithDialect, func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 1);

  for (size_t i = 2; i <= 120; ++i) {
    for (int b = 0; b < 10; ++b) {
      std::string bitstring;
      for (size_t j = 0; j < i; ++j) {
        bitstring += static_cast<bool>(dis(gen)) ? '1' : '0';
      }

      auto mod =
          groverDecomposed(&context, static_cast<int64_t>(i), 10000, bitstring);
      writeMLIR(*mod, outputDir + "/mlir/" + "grover_" + std::to_string(i) +
                          "_" + bitstring + ".mlir");
      writeQASM(*mod, outputDir + "/qasm/" + "grover_" + std::to_string(i) +
                          "_" + bitstring + ".qasm");
    }
  }

  return 0;
}
