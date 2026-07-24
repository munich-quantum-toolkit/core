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
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
// IWYU pragma: end_keep

using namespace mlir;
using namespace mlir::qco;

static bool isQCOLinearType(Type type) {
  if (isa<QubitType>(type)) {
    return true;
  }
  const auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getRank() == 1 &&
         isa<QubitType>(tensorType.getElementType());
}

//===----------------------------------------------------------------------===//
// Custom Parsers
//===----------------------------------------------------------------------===//

static ParseResult
parseTargetAliasing(OpAsmParser& parser, Region& region,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands) {
  return utils::parseTargetAliasing<QubitType>(parser, region, operands);
}

static void printTargetAliasing(OpAsmPrinter& printer, Operation* /*op*/,
                                Region& region, OperandRange targetsIn) {
  utils::printTargetAliasing(printer, region, targetsIn);
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

  SmallVector<OpAsmParser::Argument> thenArgs;
  SmallVector<OpAsmParser::Argument> elseRegionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> thenOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> elseOperands;

  if (parser.parseKeyword("args")) {
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
  const auto numLinearResults = thenOperands.size();
  if (result.types.size() < numLinearResults) {
    return parser.emitError(
        parser.getCurrentLocation(),
        "expected at least one linear result type per assigned argument");
  }
  const auto numClassicalResults = result.types.size() - numLinearResults;
  const auto linearResultTypes =
      ArrayRef(result.types).take_back(numLinearResults);
  if (llvm::any_of(linearResultTypes,
                   [](Type type) { return !isQCOLinearType(type); })) {
    return parser.emitError(parser.getCurrentLocation(),
                            "assigned arguments must correspond to trailing "
                            "linear result types");
  }
  // Resolve the operands
  if (failed(parser.resolveOperands(thenOperands, linearResultTypes,
                                    parser.getCurrentLocation(),
                                    result.operands))) {
    return failure();
  }
  // Set the argument types
  for (auto [iterArg, type] : llvm::zip_equal(thenArgs, linearResultTypes)) {
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
  if (parser.parseKeyword("args")) {
    return failure();
  }
  // Parse the else block assignment list
  if (parser.parseAssignmentList(elseRegionArgs, elseOperands)) {
    return failure();
  }
  if (elseOperands.size() != numLinearResults) {
    return parser.emitError(
        parser.getCurrentLocation(),
        "expected the same number of linear arguments in both branches");
  }

  SmallVector<Value> resolvedElseOperands;
  // Also resolve the else operands to check if they are the same as the
  // previous operands
  if (failed(parser.resolveOperands(elseOperands, linearResultTypes,
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
  for (auto [iterArg, type] :
       llvm::zip_equal(elseRegionArgs, linearResultTypes)) {
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

  llvm::copy(
      ArrayRef<int32_t>({static_cast<int32_t>(numClassicalResults),
                         static_cast<int32_t>(numLinearResults)}),
      result.getOrAddProperties<IfOp::Properties>().resultSegmentSizes.begin());

  return success();
}

void IfOp::print(OpAsmPrinter& p) {
  p << " ";
  p.printOperand(getCondition());
  // Print assignment list
  auto printQubitsBlock = [&](Region& region, OperandRange operands) {
    p << " args(";
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
  llvm::interleaveComma(getResultTypes(), p, [&](Type t) { p.printType(t); });
  p << ") ";
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false);

  // Print else region
  p << " else";
  printQubitsBlock(getElseRegion(), getQubits());
  p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false);

  p.printOptionalAttrDict((*this)->getAttrs(), {"resultSegmentSizes"});
}

LogicalResult YieldOp::verify() {
  SmallVector<Type> expectedTypes;
  bool validParent = true;
  TypeSwitch<Operation*>(getOperation()->getParentOp())
      .Case<IfOp, IndexSwitchOp, InvOp, PowOp>([&](auto parent) {
        llvm::append_range(expectedTypes, parent.getResultTypes());
      })
      .Case<CtrlOp>([&](CtrlOp parent) {
        llvm::append_range(expectedTypes, parent.getTargetsOut().getTypes());
      })
      .Default([&](Operation*) { validParent = false; });
  if (!validParent) {
    return emitOpError("has an unsupported parent operation");
  }

  if (getTargets().size() != expectedTypes.size()) {
    return emitOpError() << "must yield " << expectedTypes.size()
                         << " values for parent operation but yields "
                         << getTargets().size();
  }

  for (const auto [index, types] : llvm::enumerate(llvm::zip_equal(
           getTargets().getTypes(), TypeRange(expectedTypes)))) {
    const auto [actual, expected] = types;
    if (actual != expected) {
      return emitOpError() << "operand " << index << " has type " << actual
                           << " but parent operation expects " << expected;
    }
  }
  return success();
}

ParseResult IndexSwitchOp::parse(::mlir::OpAsmParser& parser,
                                 ::mlir::OperationState& result) {
  OpAsmParser::UnresolvedOperand index;
  if (parser.parseOperand(index) ||
      parser.resolveOperand(index, parser.getBuilder().getIndexType(),
                            result.operands)) {
    return failure();
  }

  if (parser.parseOptionalArrowTypeList(result.types)) {
    return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Create default region here to ensure regions(0) = default.
  Region* defaultRegion = result.addRegion();

  SmallVector<Value> operands;
  SmallVector<int64_t> caseValues;
  SmallVector<OpAsmParser::Argument> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> regionOperands;

  while (succeeded(parser.parseOptionalKeyword("case"))) {
    int64_t caseValue = 0;
    if (parser.parseInteger(caseValue)) {
      return failure();
    }

    caseValues.push_back(caseValue);

    if (parser.parseKeyword("args")) {
      return failure();
    }

    if (parser.parseAssignmentList(regionArgs, regionOperands)) {
      return failure();
    }

    if (caseValues.size() == 1) {

      // Resolve the operands into the result for the very first case.

      if (parser.resolveOperands(regionOperands, result.types,
                                 parser.getCurrentLocation(),
                                 result.operands)) {
        return failure();
      }
    } else {

      // Otherwise, verify if the other cases use the equivalent operands (minus
      // the case-value) as the first one.

      SmallVector<Value> operands;
      if (parser.resolveOperands(regionOperands, result.types,
                                 parser.getCurrentLocation(), operands)) {
        return failure();
      }

      for (auto [v0, v1] :
           llvm::zip_equal(operands, llvm::drop_begin(result.operands, 1))) {
        if (v0 != v1) {
          return parser.emitError(
              parser.getCurrentLocation(),
              "else qubits must reference the same SSA values as then qubits");
        }
      }
    }

    for (auto [arg, type] : llvm::zip_equal(regionArgs, result.types)) {
      arg.type = type;
    }

    if (parser.parseRegion(*result.addRegion(), regionArgs)) {
      return failure();
    }

    operands.clear();
    regionArgs.clear();
    regionOperands.clear();
  }

  result.addAttribute("cases",
                      DenseI64ArrayAttr::get(parser.getContext(), caseValues));

  // Parse the default regions and again verify if the default case uses the
  // equivalent operands (minus the case-value) as all other cases.

  if (parser.parseKeyword("default")) {
    return failure();
  }

  if (parser.parseKeyword("args")) {
    return failure();
  }

  if (parser.parseAssignmentList(regionArgs, regionOperands)) {
    return failure();
  }

  // Otherwise, verify if the other cases use the equivalent operands (minus
  // the case-value) for all other cases.

  if (parser.resolveOperands(regionOperands, result.types,
                             parser.getCurrentLocation(), operands)) {
    return failure();
  }

  if (caseValues.empty()) {
    result.operands.append(operands);
  } else {
    for (auto [v0, v1] :
         llvm::zip_equal(operands, llvm::drop_begin(result.operands, 1))) {
      if (v0 != v1) {
        return parser.emitError(
            parser.getCurrentLocation(),
            "else qubits must reference the same SSA values as then qubits");
      }
    }
  }

  for (auto [args, type] : llvm::zip_equal(regionArgs, result.types)) {
    args.type = type;
  }

  if (parser.parseRegion(*defaultRegion, regionArgs)) {
    return failure();
  }

  return success();
}

void IndexSwitchOp::print(OpAsmPrinter& p) {
  p << " ";
  p.printOperand(getArg());

  // Print result types if present
  if (!getResults().empty()) {
    p << " -> ";
    llvm::interleaveComma(getResultTypes(), p);
  }

  // Print attributes (excluding cases which we handle specially)
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     /*elidedAttrs=*/{"cases"});

  // Print case regions
  const auto cases = getCases();
  for (size_t i = 0; i < getNumCases(); ++i) {
    p.printNewline();
    p << "case ";
    p << cases[i];
    p << " args(";

    auto& region = getCaseRegions()[i];
    auto& block = region.front();

    // Print block arguments with their corresponding target operands
    for (size_t j = 0; j < block.getNumArguments(); ++j) {
      if (j > 0) {
        p << ", ";
      }
      p.printOperand(block.getArgument(j));
      p << " = ";
      p.printOperand(getTargets()[j]);
    }
    p << ") ";
    p.printRegion(region, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }

  p.printNewline();
  p << "default args(";
  auto& defaultRegion = getDefaultRegion();
  auto& defaultBlock = defaultRegion.front();

  // Print block arguments with their corresponding target operands
  for (size_t j = 0; j < defaultBlock.getNumArguments(); ++j) {
    if (j > 0) {
      p << ", ";
    }
    p.printOperand(defaultBlock.getArgument(j));
    p << " = ";
    p.printOperand(getTargets()[j]);
  }
  p << ") ";
  p.printRegion(defaultRegion, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
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
