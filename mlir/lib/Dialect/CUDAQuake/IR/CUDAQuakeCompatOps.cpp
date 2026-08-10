/*
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Dialect/CUDAQuake/IR/CUDAQuakeCompatOps.h"

#include "mlir/Dialect/CUDAQuake/IR/CUDAQuakeCompat.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h> // IWYU pragma: keep
#include <llvm/ADT/TypeSwitch.h>  // IWYU pragma: keep
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h> // IWYU pragma: keep
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <optional>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Dialects
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CUDAQuake/IR/QuakeCompatOpsDialect.cpp.inc"

void cudaq_compat::quake::QuakeCompatDialect::
    initialize() { // NOLINT(readability-convert-member-functions-to-static)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/CUDAQuake/IR/QuakeCompatOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/CUDAQuake/IR/QuakeCompatOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/CUDAQuake/IR/QuakeCompatOpsTypes.cpp.inc"

void cudaq_compat::quake::VeqType::print(AsmPrinter& printer) const {
  printer << '<';
  if (hasSpecifiedSize()) {
    printer << getSize();
  } else {
    printer << '?';
  }
  printer << '>';
}

Type cudaq_compat::quake::VeqType::parse(AsmParser& parser) {
  if (failed(parser.parseLess())) {
    return {};
  }
  uint64_t size = kDynamicSize;
  if (failed(parser.parseOptionalQuestion())) {
    if (failed(parser.parseInteger(size))) {
      return {};
    }
  }
  if (failed(parser.parseGreater())) {
    return {};
  }
  return get(parser.getContext(), size);
}

//===----------------------------------------------------------------------===//
// Quake operations
//===----------------------------------------------------------------------===//

static ParseResult
parseRawIndex(OpAsmParser& parser,
              std::optional<OpAsmParser::UnresolvedOperand>& index,
              IntegerAttr& rawIndex) {
  uint64_t constantIndex =
      cudaq_compat::quake::QuakeExtractRefOp::kDynamicIndex;
  if (const auto parsedInteger = parser.parseOptionalInteger(constantIndex);
      parsedInteger.has_value()) {
    if (failed(*parsedInteger)) {
      return failure();
    }
    index = std::nullopt;
  } else {
    OpAsmParser::UnresolvedOperand operand;
    if (failed(parser.parseOperand(operand))) {
      return failure();
    }
    index = operand;
  }
  rawIndex = IntegerAttr::get(IntegerType::get(parser.getContext(), 64),
                              llvm::APInt(64, constantIndex));
  return success();
}

static void printRawIndex(OpAsmPrinter& printer, Operation* /*op*/, Value index,
                          const IntegerAttr rawIndex) {
  if (rawIndex.getValue() ==
      cudaq_compat::quake::QuakeExtractRefOp::kDynamicIndex) {
    printer.printOperand(index);
  } else {
    printer << rawIndex.getValue();
  }
}

void cudaq_compat::quake::QuakeApplyOp::print(OpAsmPrinter& printer) {
  if (getIsAdj()) {
    printer << "<adj>";
  }
  printer << ' ';
  printer.printAttributeWithoutType(getCalleeAttr());
  if (!getControls().empty()) {
    printer << " [" << getControls() << ']';
  }
  if (!getActuals().empty()) {
    printer << ' ' << getActuals();
  }
  printer << " : ";
  printer.printFunctionalType(getOperandTypes(), getResultTypes());
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {getCalleeAttrName(), getIsAdjAttrName(),
                                 getOperandSegmentSizesAttrName()});
}

ParseResult cudaq_compat::quake::QuakeApplyOp::parse(OpAsmParser& parser,
                                                     OperationState& state) {
  if (succeeded(parser.parseOptionalLess())) {
    if (failed(parser.parseKeyword("adj")) || failed(parser.parseGreater())) {
      return failure();
    }
    state.addAttribute("is_adj", parser.getBuilder().getUnitAttr());
  }

  SymbolRefAttr callee;
  NamedAttrList calleeAttrs;
  if (failed(parser.parseCustomAttributeWithFallback(
          callee, parser.getBuilder().getNoneType(), "callee", calleeAttrs))) {
    return failure();
  }
  state.addAttribute("callee", callee);

  SmallVector<OpAsmParser::UnresolvedOperand> controls;
  if (succeeded(parser.parseOptionalLSquare()) &&
      (failed(parser.parseOperandList(controls)) ||
       failed(parser.parseRSquare()))) {
    return failure();
  }
  SmallVector<OpAsmParser::UnresolvedOperand> actuals;
  if (failed(parser.parseOperandList(actuals)) || failed(parser.parseColon())) {
    return failure();
  }

  FunctionType functionType;
  if (failed(parser.parseType(functionType)) ||
      failed(parser.parseOptionalAttrDict(state.attributes))) {
    return failure();
  }
  if (functionType.getNumInputs() != controls.size() + actuals.size()) {
    return parser.emitError(parser.getNameLoc(),
                            "operand count does not match function type");
  }
  SmallVector<OpAsmParser::UnresolvedOperand> operands(controls);
  operands.append(actuals);
  if (failed(parser.resolveOperands(operands, functionType.getInputs(),
                                    parser.getNameLoc(), state.operands))) {
    return failure();
  }
  state.addTypes(functionType.getResults());
  state.addAttribute("operand_segment_sizes",
                     parser.getBuilder().getDenseI32ArrayAttr(
                         {static_cast<int32_t>(controls.size()),
                          static_cast<int32_t>(actuals.size())}));
  return success();
}

//===----------------------------------------------------------------------===//
// Generated operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/CUDAQuake/IR/QuakeCompatOps.cpp.inc"
