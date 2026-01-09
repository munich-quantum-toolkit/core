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

namespace mlir::qco {
ParseResult
parseTargetAliasing(OpAsmParser& parser, Region& region,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands) {
  // 1. Parse the opening parenthesis
  if (parser.parseLParen())
    return failure();

  // Temporary storage for block arguments we are about to create
  SmallVector<OpAsmParser::Argument> blockArgs;

  // 2. Prepare to parse the list
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::Argument newArg;              // The "new" variable name
      OpAsmParser::UnresolvedOperand oldOperand; // The "old" input variable

      // Parse "%new"
      if (parser.parseArgument(newArg))
        return failure();

      // Parse "="
      if (parser.parseEqual())
        return failure();

      // Parse "%old"
      if (parser.parseOperand(oldOperand))
        return failure();
      operands.push_back(oldOperand);

      Type type = qco::QubitType::get(parser.getBuilder().getContext());
      newArg.type = type;
      blockArgs.push_back(newArg);

    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen())
      return failure();
  }

  // 4. Parse the Region
  // We explicitly pass the blockArgs we just parsed so they become the entry
  // block!
  if (parser.parseRegion(region, blockArgs))
    return failure();

  return success();
}

void printTargetAliasing(OpAsmPrinter& printer, Operation* op, Region& region,
                         OperandRange targets_in) {
  printer << "(";
  Block& entryBlock = region.front();
  auto blockArgs = entryBlock.getArguments();

  for (unsigned i = 0; i < targets_in.size(); ++i) {
    if (i > 0)
      printer << ", ";
    printer.printOperand(blockArgs[i]);
    printer << " = ";
    printer.printOperand(targets_in[i]);
  }
  printer << ") ";

  printer.printRegion(region, /*printEntryBlockArgs=*/false);
}
} // namespace mlir::qco

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
