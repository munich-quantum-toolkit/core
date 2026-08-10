/*
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Dialect/CUDAQuake/IR/CUDAQuakeCompat.h"
#include "mlir/Dialect/CUDAQuake/IR/CUDAQuakeCompatOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h> // IWYU pragma: keep
#include <llvm/ADT/TypeSwitch.h>  // IWYU pragma: keep
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h> // IWYU pragma: keep
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;

#include "mlir/Dialect/CUDAQuake/IR/CCCompatOpsDialect.cpp.inc"

void cudaq_compat::cc::CCCompatDialect::
    initialize() { // NOLINT(readability-convert-member-functions-to-static)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/CUDAQuake/IR/CCCompatOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/CUDAQuake/IR/CCCompatOps.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/CUDAQuake/IR/CCCompatOpsTypes.cpp.inc"

static void ensureContinueTerminator(OpBuilder& builder, OperationState& state,
                                     Region* region) {
  if (region->empty()) {
    return;
  }
  Block& block = region->back();
  if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>()) {
    return;
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&block);
  cudaq_compat::cc::CCContinueOp::create(builder, state.location, ValueRange{});
}

void cudaq_compat::cc::CCScopeOp::print(OpAsmPrinter& printer) {
  const bool printTerminators =
      !getBody().hasOneBlock() || getNumResults() != 0;
  printer.printOptionalArrowTypeList(getResultTypes());
  printer << ' ';
  printer.printRegion(getBody(), false, printTerminators);
  printer.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult cudaq_compat::cc::CCScopeOp::parse(OpAsmParser& parser,
                                               OperationState& state) {
  auto* body = state.addRegion();
  if (failed(parser.parseOptionalArrowTypeList(state.types)) ||
      failed(parser.parseRegion(*body)) ||
      failed(parser.parseOptionalAttrDict(state.attributes))) {
    return failure();
  }
  OpBuilder builder(parser.getContext());
  ensureContinueTerminator(builder, state, body);
  return success();
}

void cudaq_compat::cc::CCCreateLambdaOp::print(OpAsmPrinter& printer) {
  printer << ' ';
  printer.printRegion(getBody(), !getBody().getArguments().empty(), true);
  printer << " : " << getCallable().getType();
  printer.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult cudaq_compat::cc::CCCreateLambdaOp::parse(OpAsmParser& parser,
                                                      OperationState& state) {
  auto* body = state.addRegion();
  Type callableType;
  if (failed(parser.parseRegion(*body)) ||
      failed(parser.parseColonType(callableType)) ||
      failed(parser.parseOptionalAttrDict(state.attributes))) {
    return failure();
  }
  if (!isa<cudaq_compat::cc::CallableType>(callableType)) {
    return parser.emitError(parser.getNameLoc(),
                            "expected a !cc.callable result type");
  }
  state.addTypes(callableType);
  return success();
}

void cudaq_compat::cc::CCIfOp::print(OpAsmPrinter& printer) {
  printer << '(' << getCondition() << ')';
  printer.printOptionalArrowTypeList(getResultTypes());
  printer << ' ';
  const bool printTerminators =
      !getThenRegion().hasOneBlock() || getNumResults() != 0;
  printer.printRegion(getThenRegion(), false, printTerminators);
  if (!getElseRegion().empty()) {
    printer << " else ";
    const bool printElseTerminators =
        !getElseRegion().hasOneBlock() || getNumResults() != 0;
    printer.printRegion(getElseRegion(), false, printElseTerminators);
  }
  printer.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult cudaq_compat::cc::CCIfOp::parse(OpAsmParser& parser,
                                            OperationState& state) {
  auto* thenRegion = state.addRegion();
  auto* elseRegion = state.addRegion();
  OpAsmParser::UnresolvedOperand condition;
  if (failed(parser.parseLParen()) || failed(parser.parseOperand(condition)) ||
      failed(parser.parseRParen()) ||
      failed(parser.resolveOperand(condition, parser.getBuilder().getI1Type(),
                                   state.operands)) ||
      failed(parser.parseOptionalArrowTypeList(state.types)) ||
      failed(parser.parseRegion(*thenRegion))) {
    return failure();
  }

  OpBuilder builder(parser.getContext());
  ensureContinueTerminator(builder, state, thenRegion);
  if (succeeded(parser.parseOptionalKeyword("else"))) {
    if (failed(parser.parseRegion(*elseRegion))) {
      return failure();
    }
    ensureContinueTerminator(builder, state, elseRegion);
  }
  return parser.parseOptionalAttrDict(state.attributes);
}

static void printInitializationList(OpAsmPrinter& printer,
                                    Block::BlockArgListType blockArgs,
                                    Operation::operand_range initializers) {
  if (initializers.empty()) {
    return;
  }
  printer << "((";
  llvm::interleaveComma(
      llvm::zip(blockArgs, initializers), printer, [&](const auto item) {
        printer << std::get<0>(item) << " = " << std::get<1>(item);
      });
  printer << ") -> (" << initializers.getTypes() << ")) ";
}

void cudaq_compat::cc::CCLoopOp::print(OpAsmPrinter& printer) {
  if (getPostCondition()) {
    printer << " do ";
    printInitializationList(printer, getBodyRegion().front().getArguments(),
                            getInitialArgs());
    printer.printRegion(getBodyRegion(), false, true);
    printer << " while ";
    printer.printRegion(getWhileRegion(), !getInitialArgs().empty(), true);
  } else {
    printer << " while ";
    printInitializationList(printer, getWhileRegion().front().getArguments(),
                            getInitialArgs());
    printer.printRegion(getWhileRegion(), false, true);
    printer << " do ";
    printer.printRegion(getBodyRegion(), !getInitialArgs().empty(), true);
    if (!getStepRegion().empty()) {
      printer << " step ";
      printer.printRegion(getStepRegion(), !getInitialArgs().empty(), true);
    }
    if (!getElseRegion().empty()) {
      printer << " else ";
      printer.printRegion(getElseRegion(), !getInitialArgs().empty(), true);
    }
  }
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {getPostConditionAttrName()});
}

ParseResult cudaq_compat::cc::CCLoopOp::parse(OpAsmParser& parser,
                                              OperationState& state) {
  auto* whileRegion = state.addRegion();
  auto* bodyRegion = state.addRegion();
  auto* stepRegion = state.addRegion();
  auto* elseRegion = state.addRegion();
  bool postCondition = false;

  const auto parseInitializers =
      [&](SmallVectorImpl<OpAsmParser::Argument>& regionArgs) -> ParseResult {
    if (failed(parser.parseOptionalLParen())) {
      return success();
    }
    SmallVector<OpAsmParser::UnresolvedOperand> operands;
    if (failed(parser.parseAssignmentList(regionArgs, operands)) ||
        failed(parser.parseArrowTypeList(state.types)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
    for (auto [argument, operand, type] :
         llvm::zip(regionArgs, operands, state.types)) {
      argument.type = type;
      if (failed(parser.resolveOperand(operand, type, state.operands))) {
        return failure();
      }
    }
    return success();
  };

  if (succeeded(parser.parseOptionalKeyword("while"))) {
    SmallVector<OpAsmParser::Argument> regionArgs;
    if (failed(parseInitializers(regionArgs)) ||
        failed(parser.parseRegion(*whileRegion, regionArgs)) ||
        failed(parser.parseKeyword("do")) ||
        failed(parser.parseRegion(*bodyRegion))) {
      return failure();
    }
    if (succeeded(parser.parseOptionalKeyword("step")) &&
        failed(parser.parseRegion(*stepRegion))) {
      return failure();
    }
    if (succeeded(parser.parseOptionalKeyword("else")) &&
        failed(parser.parseRegion(*elseRegion))) {
      return failure();
    }
  } else if (succeeded(parser.parseOptionalKeyword("do"))) {
    postCondition = true;
    SmallVector<OpAsmParser::Argument> regionArgs;
    if (failed(parseInitializers(regionArgs)) ||
        failed(parser.parseRegion(*bodyRegion, regionArgs)) ||
        failed(parser.parseKeyword("while")) ||
        failed(parser.parseRegion(*whileRegion))) {
      return failure();
    }
  } else {
    return parser.emitError(parser.getNameLoc(), "expected 'while' or 'do'");
  }

  state.addAttribute("post_condition",
                     parser.getBuilder().getBoolAttr(postCondition));
  return parser.parseOptionalAttrDict(state.attributes);
}

#define GET_OP_CLASSES
#include "mlir/Dialect/CUDAQuake/IR/CCCompatOps.cpp.inc"
