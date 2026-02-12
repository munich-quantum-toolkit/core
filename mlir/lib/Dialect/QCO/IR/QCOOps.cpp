/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h" // IWYU pragma: associated

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
// IWYU pragma: end_keep

using namespace mlir;
using namespace mlir::qco;

//===----------------------------------------------------------------------===//
// Custom Parsers
//===----------------------------------------------------------------------===//

static ParseResult
parseTargetAliasing(OpAsmParser& parser, Region& region,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands) {
  // 1. Parse the opening parenthesis
  if (parser.parseLParen()) {
    return failure();
  }

  // Temporary storage for block arguments we are about to create
  SmallVector<OpAsmParser::Argument> blockArgs;

  // 2. Prepare to parse the list
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::Argument newArg;              // The "new" variable name
      OpAsmParser::UnresolvedOperand oldOperand; // The "old" input variable

      // Parse "%new"
      if (parser.parseArgument(newArg)) {
        return failure();
      }

      // Parse "="
      if (parser.parseEqual()) {
        return failure();
      }

      // Parse "%old"
      if (parser.parseOperand(oldOperand)) {
        return failure();
      }
      operands.push_back(oldOperand);

      // Hard-code QubitType since targets in qco.ctrl are always qubits.
      // This avoids double-binding type($targets_in) in the assembly format
      // while keeping the parser simple and the assembly format clean.
      newArg.type = QubitType::get(parser.getBuilder().getContext());
      blockArgs.push_back(newArg);

    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen()) {
      return failure();
    }
  }

  // 4. Parse the Region
  // We explicitly pass the blockArgs we just parsed so they become the entry
  // block!
  if (parser.parseRegion(region, blockArgs)) {
    return failure();
  }

  return success();
}

static void printTargetAliasing(OpAsmPrinter& printer, Operation* /*op*/,
                                Region& region, OperandRange targetsIn) {
  printer << "(";
  if (region.empty()) {
    printer << ") ";
    printer.printRegion(region, false);
    return;
  }
  Block& entryBlock = region.front();

  const auto numTargets = targetsIn.size();
  for (unsigned i = 0; i < numTargets; ++i) {
    if (i > 0) {
      printer << ", ";
    }
    printer.printOperand(entryBlock.getArgument(i));
    printer << " = ";
    printer.printOperand(targetsIn[i]);
  }
  printer << ") ";

  printer.printRegion(region, false);
}

static ParseResult
parseIfOpAliasing(OpAsmParser& parser, Region& thenRegion, Region& elseRegion,
                  SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands) {
  // Parse the qubits keyword
  if (parser.parseKeyword("qubits")) {
    return failure();
  }

  // Parse the then region
  if (parseTargetAliasing(parser, thenRegion, operands)) {
    return failure();
  }

  // Parse the else keyword
  if (parser.parseKeyword("else")) {
    return failure();
  }

  // Parse the qubits keyword
  if (parser.parseKeyword("qubits")) {
    return failure();
  }

  // Parse the else region
  if (parseTargetAliasing(parser, elseRegion, operands)) {
    return failure();
  }

  // Remove duplicate operands
  llvm::DenseSet<llvm::StringRef> seen;
  llvm::erase_if(operands,
                 [&](const auto& op) { return !seen.insert(op.name).second; });

  return success();
}

static void printIfOpAliasing(OpAsmPrinter& printer, Operation* op,
                              Region& thenRegion, Region& elseRegion,
                              OperandRange qubits) {
  printer << "qubits";
  printTargetAliasing(printer, op, thenRegion, qubits);
  printer << " else ";
  printer << "qubits";
  printTargetAliasing(printer, op, elseRegion, qubits);
}

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QCO/IR/QCOOpsDialect.cpp.inc"

void QCODialect::initialize() {
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/QCO/IR/QCOOpsTypes.cpp.inc"

      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/QCO/IR/QCOOps.cpp.inc"

      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/QCO/IR/QCOOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QCO/IR/QCOInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/QCO/IR/QCOOps.cpp.inc"
