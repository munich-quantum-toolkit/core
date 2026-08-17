/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Generate.h"

#include "mlir/Benchmark/Programs.h"
#include "mlir/Compiler/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <jeff/IR/JeffDialect.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace mqt::benchmarks {

using namespace mlir;

std::optional<JeffProgram> buildJeffProgram(const Benchmark& benchmark,
                                            const uint64_t n) {
  if (n < benchmark.minimumSize) {
    return std::nullopt;
  }

  DialectRegistry registry;
  // The conversions to QCO and jeff create operations from these dialects, so
  // every one of them must be loaded before the pipeline runs.
  registry
      .insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
              arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
              scf::SCFDialect, LLVM::LLVMDialect, memref::MemRefDialect,
              tensor::TensorDialect, jeff::JeffDialect>();
  auto context = std::make_shared<MLIRContext>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  qc::QCProgramBuilder builder(context.get());
  builder.initialize();
  auto results = benchmark.build(builder, n);

  // `initialize` defaults the entry point to an integer result, so the function
  // is retyped to the classical registers the program returns.
  SmallVector<Type> resultTypes;
  resultTypes.reserve(results.size());
  for (auto result : results) {
    resultTypes.emplace_back(result.getType());
  }
  builder.retype(resultTypes);

  auto moduleOp = builder.finalize(results);
  if (!moduleOp) {
    return std::nullopt;
  }

  auto program = QCProgram::fromModule(context, std::move(moduleOp));
  if (!program || !program->cleanup()) {
    return std::nullopt;
  }

  auto qco = std::move(*program).intoQCO();
  if (!qco) {
    return std::nullopt;
  }
  // `jeff` represents a modifier as attributes on a single gate, so modifiers
  // that wrap several operations are unrolled first. The optimization pipeline
  // can empty a modifier body, for example when it folds a zero-angle rotation
  // away, so the cleanup runs afterwards to erase the modifiers left behind.
  if (!qco->runPassPipeline("unroll-modifiers") ||
      !qco->runPassPipeline("mqt-qco-default") || !qco->cleanup()) {
    return std::nullopt;
  }

  return std::move(*qco).intoJeff();
}

} // namespace mqt::benchmarks
