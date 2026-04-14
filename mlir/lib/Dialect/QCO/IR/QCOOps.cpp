/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include "mlir/Dialect/QCO/IR/QCODialect.h" // IWYU pragma: associated

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
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

ParseResult IfOp::parse(::mlir::OpAsmParser& parser,
                        ::mlir::OperationState& result) {
  auto& builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  auto i1Type = builder.getIntegerType(1);
  // Resolve the condition operand
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands)) {
    return failure();
  }

  SmallVector<OpAsmParser::Argument, 4> thenArgs;
  SmallVector<OpAsmParser::Argument, 4> elseRegionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> thenOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> elseOperands;

  if (parser.parseKeyword("qubits")) {
    return failure();
  }
  // Parse the then block assignment list
  if (parser.parseAssignmentList(thenArgs, thenOperands)) {
    return failure();
  }
  // Parse result type list
  if (parser.parseArrowTypeList(result.types)) {
    return failure();
  }
  // Resolve the operands
  if (failed(parser.resolveOperands(thenOperands, result.types,
                                    parser.getCurrentLocation(),
                                    result.operands))) {
    return failure();
  }
  // Set the argument types
  for (auto [iterArg, type] : llvm::zip_equal(thenArgs, result.types)) {
    iterArg.type = type;
  }
  // Parse the then region
  Region* body = result.addRegion();
  if (parser.parseRegion(*body, thenArgs)) {
    return failure();
  }
  if (parser.parseKeyword("else")) {
    return failure();
  }
  if (parser.parseKeyword("qubits")) {
    return failure();
  }
  // Parse the else block assignment list
  if (parser.parseAssignmentList(elseRegionArgs, elseOperands)) {
    return failure();
  }

  SmallVector<Value> resolvedElseOperands;
  // Also resolve the else operands to check if they are the same as the
  // previous operands
  if (failed(parser.resolveOperands(elseOperands, result.types,
                                    parser.getCurrentLocation(),
                                    resolvedElseOperands))) {
    return failure();
  }
  for (auto [elseVal, thenVal] : llvm::zip_equal(
           resolvedElseOperands, llvm::drop_begin(result.operands,
                                                  1))) { // skip condition
    if (elseVal != thenVal) {
      return parser.emitError(
          parser.getCurrentLocation(),
          "else qubits must reference the same SSA values as then qubits");
    }
  }

  // Set the argument types
  for (auto [iterArg, type] : llvm::zip_equal(elseRegionArgs, result.types)) {
    iterArg.type = type;
  }
  // Parse the else region
  Region* elseBody = result.addRegion();
  if (parser.parseRegion(*elseBody, elseRegionArgs)) {
    return failure();
  };

  // Parse optional attr
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

void IfOp::print(OpAsmPrinter& p) {
  p << " ";
  p.printOperand(getCondition());
  // Print assignment list
  auto printQubitsBlock = [&](Region& region, OperandRange operands) {
    p << " qubits(";
    if (!region.empty()) {
      Block& entry = region.front();
      llvm::interleaveComma(
          llvm::zip(entry.getArguments(), operands), p, [&](auto pair) {
            p.printRegionArgument(std::get<0>(pair), /*attrs=*/{},
                                  /*omitType=*/true);
            p << " = ";
            p.printOperand(std::get<1>(pair));
          });
    }
    p << ") ";
  };
  // Print then region
  printQubitsBlock(getThenRegion(), getQubits());
  // Print result types
  p << "-> (";
  llvm::interleaveComma(getThenRegion().front().getArgumentTypes(), p,
                        [&](Type t) { p.printType(t); });
  p << ") ";
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false);

  // Print else region
  p << " else";
  printQubitsBlock(getElseRegion(), getQubits());
  p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false);

  p.printOptionalAttrDict((*this)->getAttrs());
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
