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
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQCToOpenQASM3.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
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
#include <mlir/IR/ValueRange.h>
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

static OwningOpRef<ModuleOp> prepare(OwningOpRef<ModuleOp> mod) {
  PassManager pm(mod->getContext());
  pm.addPass(createDecomposeMultiControlled());
  pm.addPass(createCanonicalizerPass());
  pm.run(*mod);
  return mod;
}

/// Return a structured program implementing Grover's algorithm using
/// @p numQubits qubits.
static OwningOpRef<ModuleOp> grover(MLIRContext* context, const int64_t nqubits,
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

  return builder.finalize(bits);
}

static OwningOpRef<ModuleOp> vqe(MLIRContext* context, const int64_t nqubits,
                                 const int64_t nlayers, const double decay) {
  /// The angle the optimizer starts from.
  constexpr double vqeInitialAngle = llvm::numbers::pi / 2.0;

  QCOProgramBuilder builder(context);
  builder.initialize(SmallVector<Type>{builder.getF64Type()});

  SmallVector<Value> bits(nqubits);
  SmallVector<Value> qubits(nqubits);

  auto tensor = builder.qtensorAlloc(nqubits);
  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }

  // A round prepares the ansatz at the current angle, reads the register, and
  // estimates the energy of an Ising chain from the measured bits. The
  // optimizer shrinks the angle and runs another round only while the energy
  // improves on the round before it, so the rounds follow from the
  // measurements.

  SmallVector<Value> inArgs{
      /* improved = */ builder.boolConstant(false),
      /* angle = */ builder.floatConstant(vqeInitialAngle),
      /* previous = */ builder.intConstant(1)};
  inArgs.append(qubits);

  auto outArgs = builder.scfWhile(
      inArgs,
      [&](ValueRange args) {
        SmallVector<Value> bodyArgs(args);
        auto& improved = bodyArgs[0];
        builder.scfCondition(improved, bodyArgs);
        return bodyArgs;
      },
      [&](ValueRange args) {
        SmallVector<Value> bodyArgs(args);

        auto& improved = bodyArgs[0];
        auto& angle = bodyArgs[1];
        auto& previous = bodyArgs[2];
        SmallVector<Value> bodyQubits(ArrayRef(bodyArgs).drop_front(3));

        // TODO Reset
        for_each(bodyQubits, [&](auto& q) { q = builder.h(q); });

        bodyQubits = builder.scfFor(
            0, nlayers, 1, bodyQubits, [&](Value, ValueRange forArgs) {
              SmallVector<Value> forBodyQubits(forArgs);
              for_each(forBodyQubits,
                       [&](auto& q) { q = builder.ry(angle, q); });

              for (size_t i = 0; i < nqubits - 1; ++i) {
                std::tie(forBodyQubits[i], forBodyQubits[i + 1]) =
                    builder.cx(forBodyQubits[i], forBodyQubits[i + 1]);
              }
              return forBodyQubits;
            });

        bodyQubits = builder.barrier(bodyQubits);

        SmallVector<Value> bits(bodyQubits.size());
        for (int64_t i = 0; i < bodyQubits.size(); ++i) {
          std::tie(bodyQubits[i], bits[i]) = builder.measure(bodyQubits[i]);
        }

        for (size_t i = 3; i < bodyArgs.size(); ++i) {
          bodyArgs[i] = bodyQubits[i - 3];
        }

        // Compute energy:
        // The energy of the chain counts the neighbouring pairs that disagree.
        auto energy =
            arith::ExtUIOp::create(builder, builder.getI64Type(), bits[0])
                .getResult();

        // Update angle.
        angle =
            arith::MulFOp::create(builder, angle, builder.floatConstant(decay))
                .getResult();
        // Compute exit condition.
        improved = arith::CmpIOp::create(builder, arith::CmpIPredicate::slt,
                                         energy, previous);
        previous = energy;

        return bodyArgs;
      });

  qubits = builder.barrier(outArgs.drop_front(3));

  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  for (int64_t i = 0; i < nqubits; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  return builder.finalize(outArgs[1]);
}

static OwningOpRef<ModuleOp> qaoa(MLIRContext* context, const int64_t nqubits,
                                  const int64_t nlayers, const double gamma,
                                  const double beta) {
  QCOProgramBuilder builder(context);
  builder.initialize(SmallVector<Type>(nqubits, builder.getI1Type()));

  SmallVector<Value> qubits(nqubits);
  SmallVector<Value> bits(nqubits);

  Value tensor = builder.qtensorAlloc(nqubits);
  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }

  // Initialize: apply Hadamard to all qubits to create uniform superposition
  for_each(qubits, [&](auto& q) { q = builder.h(q); });

  // Each layer applies the cost operator of a ring of couplings and then the
  // mixer. The problem graph is fixed, so the layer count is a constant.

  qubits = builder.scfFor(0, nlayers, 1, qubits, [&](Value, ValueRange args) {
    SmallVector<Value> bodyQubits(args);

    for (int64_t i = 0; i < nqubits - 1; ++i) {
      std::tie(bodyQubits[i], bodyQubits[i + 1]) =
          builder.rzz(gamma, bodyQubits[i], bodyQubits[i + 1]);
    }

    std::tie(bodyQubits[nqubits - 1], bodyQubits[0]) =
        builder.rzz(beta, bodyQubits[nqubits - 1], bodyQubits[0]);

    for_each(bodyQubits, [&](auto& q) { q = builder.rx(beta, q); });

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

  return builder.finalize(bits);
}

static OwningOpRef<ModuleOp> mlqae(MLIRContext* context,
                                   const int64_t nqubits) {
  const auto innerLoop = [](QCOProgramBuilder& builder, Value iv,
                            ValueRange args) {
    constexpr double mlqaeAngle = llvm::numbers::pi / 5.0;
    SmallVector<Value> qubits(args);

    qubits.back() = builder.z(qubits.back());

    const auto out2 =
        builder.mcry(-mlqaeAngle, ArrayRef(qubits).drop_back(), qubits.back());
    for (size_t i = 0; i < qubits.size() - 1; ++i) {
      qubits[i] = out2.first[i];
    }
    qubits.back() = out2.second;
    for (size_t i = 0; i < qubits.size() - 1; ++i) {
      qubits[i] = builder.h(qubits[i]);
      qubits[i] = builder.x(qubits[i]);
    }

    qubits.back() = builder.x(qubits.back());

    const auto out3 = builder.mcz(ArrayRef(qubits).drop_back(), qubits.back());
    for (size_t i = 0; i < qubits.size() - 1; ++i) {
      qubits[i] = out3.first[i];
    }
    qubits.back() = out3.second;

    qubits.back() = builder.x(qubits.back());
    for (size_t i = 0; i < qubits.size() - 1; ++i) {
      qubits[i] = builder.x(qubits[i]);
      qubits[i] = builder.h(qubits[i]);
    }

    const auto out4 =
        builder.mcry(mlqaeAngle, ArrayRef(qubits).drop_back(), qubits.back());
    for (size_t i = 0; i < qubits.size() - 1; ++i) {
      qubits[i] = out4.first[i];
    }
    qubits.back() = out4.second;

    return qubits;
  };

  const auto outerLoop = [&](QCOProgramBuilder& builder, Value iv,
                             ValueRange args) {
    constexpr double mlqaeAngle = llvm::numbers::pi / 5.0;
    SmallVector<Value> qubits(args);

    for_each(qubits, [&](auto& q) { q = builder.reset(q); });
    for_each(qubits, [&](auto& q) { q = builder.h(q); });

    const auto out =
        builder.mcry(mlqaeAngle, ArrayRef(qubits).drop_back(), qubits.back());
    for (size_t i = 0; i < qubits.size() - 1; ++i) {
      qubits[i] = out.first[i];
    }
    qubits.back() = out.second;

    auto one = builder.indexConstant(1);
    auto power = arith::AddIOp::create(builder, one, iv);

    qubits =
        builder.scfFor(0, power, 1, qubits, [&](Value iv, ValueRange args) {
          return innerLoop(builder, iv, args);
        });

    return qubits;
  };

  QCOProgramBuilder builder(context);
  builder.initialize(SmallVector<Type>(nqubits, builder.getI1Type()));

  auto c = builder.allocClassicalBitRegister(nqubits - 1, "c");

  SmallVector<Value> qubits(nqubits);
  SmallVector<Value> bits(nqubits);

  Value one = builder.indexConstant(1);
  Value tensor = builder.qtensorAlloc(nqubits - 1);

  for (int64_t i = 0; i < nqubits - 1; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }
  qubits[nqubits - 1] = builder.allocQubit();

  qubits =
      builder.scfFor(0, nqubits - 1, 1, qubits, [&](Value iv, ValueRange args) {
        return outerLoop(builder, iv, args);
      });

  qubits = builder.barrier(qubits);

  // Measure all qubits
  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  // Clean up
  for (int64_t i = 0; i < nqubits - 1; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.sink(qubits[nqubits - 1]);

  builder.qtensorDealloc(tensor);

  return builder.finalize(bits);
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
  registry.insert<qc::QCDialect, cbit::CBitDialect, memref::MemRefDialect,
                  QCODialect, qtensor::QTensorDialect, scf::SCFDialect,
                  arith::ArithDialect, func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 1);

  SmallVector<std::pair<std::string, OwningOpRef<ModuleOp>>> programs;
  for (size_t i = 2; i <= 120; ++i) {

    // Grover

    std::string bitstring;
    for (size_t j = 0; j < i; ++j) {
      bitstring += static_cast<bool>(dis(gen)) ? '1' : '0';
    }
    programs.emplace_back(
        "grover_" + std::to_string(i),
        prepare(grover(&context, static_cast<int64_t>(i), 10000, bitstring)));

    // VQE

    programs.emplace_back(
        "vqe_" + std::to_string(i),
        prepare(vqe(&context, static_cast<int64_t>(i), 10000, 0.5)));

    // QAOA

    programs.emplace_back(
        "qaoa_" + std::to_string(i),
        prepare(qaoa(&context, static_cast<int64_t>(i), 10000, 0.5, 0.1)));
  }

  for (size_t i = 2; i <= 120; ++i) {
    // MLQAE

    programs.emplace_back("mlqae_" + std::to_string(i),
                          prepare(mlqae(&context, static_cast<int64_t>(i))));
  }

  for (const auto& [name, m] : programs) {
    writeMLIR(*m, outputDir + "/mlir/" + name + ".mlir");
    writeQASM(*m, outputDir + "/qasm/" + name + ".qasm");
  }

  return 0;
}
