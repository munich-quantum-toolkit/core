/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/Translation/TranslateQCToOpenQASM3.h"

#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Target/OpenQASM/GateCatalog.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/IndentedOstream.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mlir::qc {
namespace {

enum class ResourceKind : uint8_t {
  Qubit,
  Bit,
};

struct Resource {
  ResourceKind kind;
  std::string name;
  int64_t width = 1;
  bool scalar = false;
  bool output = false;
  cbit::Initialization initialization = cbit::Initialization::Undefined;
};

struct ScalarOutput {
  Value value;
  std::string name;
  std::string kind;
};

struct GateCall {
  std::string modifiers;
  std::string symbol;
  SmallVector<std::string> parameters;
  SmallVector<std::string> qubits;
};

} // namespace

[[nodiscard]] static bool isOpenQASMIdentifier(const StringRef value) {
  if (value.empty() ||
      (!llvm::isAlpha(value.front()) && value.front() != '_')) {
    return false;
  }
  return llvm::all_of(value.drop_front(), [](const char character) {
    return llvm::isAlnum(character) || character == '_';
  });
}

[[nodiscard]] static bool isReservedOpenQASMIdentifier(const StringRef value) {
  return llvm::StringSwitch<bool>(value)
      .Cases("OPENQASM", "include", "input", "output", "const", true)
      .Cases("let", "fixed", "gate", "def", "extern", true)
      .Cases("defcalgrammar", "defcal", "cal", "opaque", "box", true)
      .Cases("delay", "reset", "measure", "barrier", true)
      .Cases("ctrl", "negctrl", "inv", "pow", true)
      .Cases("if", "else", "while", "for", "in", true)
      .Cases("break", "continue", "end", "return", true)
      .Cases("switch", "case", "default", true)
      .Cases("qubit", "qreg", "creg", "bit", "bool", true)
      .Cases("int", "uint", "float", "angle", "complex", true)
      .Cases("array", "duration", "stretch", "readonly", "mutable", true)
      .Cases("sizeof", "durationof", "true", "false", true)
      .Default(false);
}

[[nodiscard]] static bool isValidOutputName(const StringRef value) {
  return isOpenQASMIdentifier(value) && !value.starts_with("_mqt_") &&
         !isReservedOpenQASMIdentifier(value) &&
         oq3::frontend::lookupGate(value) == nullptr;
}

[[nodiscard]] static std::optional<int64_t> getConstantInteger(Value value) {
  return getConstantIntValue(value);
}

namespace {

class OpenQASMEmitter {
public:
  explicit OpenQASMEmitter(ModuleOp moduleOp) : moduleOp(moduleOp) {}

  [[nodiscard]] FailureOr<std::string> emit() {
    if (failed(verify(moduleOp)) || failed(preflight()) ||
        failed(collectProgramShape())) {
      return failure();
    }

    std::string body;
    llvm::raw_string_ostream bodyStream(body);
    raw_indented_ostream bodyOutput(bodyStream);
    output = &bodyOutput;

    if (failed(emitDeclarations()) ||
        failed(emitBlock(function.getBody().front()))) {
      return failure();
    }
    bodyOutput.flush();

    std::string source;
    llvm::raw_string_ostream sourceStream(source);
    sourceStream << "OPENQASM 3.1;\n"
                    "include \"stdgates.inc\";\n\n";
    emitFixedHelpers(sourceStream);
    for (const auto& helper : compositeHelpers) {
      sourceStream << helper << "\n";
    }
    sourceStream << body;
    return source;
  }

private:
  ModuleOp moduleOp;
  func::FuncOp function;
  raw_indented_ostream* output = nullptr;
  DenseMap<Value, Resource> resources;
  SmallVector<Value> resourceOrder;
  DenseMap<Value, std::string> valueNames;
  DenseSet<Value> returnedRegisters;
  SmallVector<ScalarOutput> scalarOutputs;
  llvm::StringSet<> usedNames;
  llvm::StringSet<> fixedHelpers;
  SmallVector<std::string> compositeHelpers;
  size_t nextQubit = 0;
  size_t nextBit = 0;
  size_t nextScalar = 0;
  size_t nextLoop = 0;
  size_t nextHelper = 0;

  [[nodiscard]] static LogicalResult fail(Operation* operation,
                                          const Twine& message) {
    operation->emitError() << "OpenQASM emission error: " << message;
    return failure();
  }

  [[nodiscard]] static FailureOr<std::string>
  failExpression(Value value, const Twine& message) {
    emitError(value.getLoc()) << "OpenQASM emission error: " << message;
    return failure();
  }

  [[nodiscard]] std::string uniqueName(const StringRef prefix,
                                       size_t& counter) {
    while (true) {
      auto candidate = (Twine("_mqt_") + prefix + Twine(counter++)).str();
      if (usedNames.insert(candidate).second) {
        return candidate;
      }
    }
  }

  [[nodiscard]] std::string outputName(const StringRef requested) {
    if (isValidOutputName(requested) && usedNames.insert(requested).second) {
      return requested.str();
    }
    return uniqueName("out", nextScalar);
  }

  [[nodiscard]] std::string qubitRegisterName(const StringRef requested) {
    if (isValidOutputName(requested) && usedNames.insert(requested).second) {
      return requested.str();
    }
    return uniqueName("q", nextQubit);
  }

  [[nodiscard]] LogicalResult preflight() {
    SmallVector<func::FuncOp> functions(moduleOp.getOps<func::FuncOp>());
    if (functions.size() != 1) {
      return fail(moduleOp, "expected exactly one function");
    }
    function = functions.front();
    if (function.isExternal() || function.getBody().getBlocks().size() != 1) {
      return fail(function,
                  "expected one defined function with one entry block");
    }
    if (function.getNumArguments() != 0) {
      return fail(function, "function arguments and OpenQASM inputs are not "
                            "supported");
    }
    const auto walkResult = function.walk([&](Operation* operation) {
      if (isa<func::CallOp>(operation)) {
        std::ignore = fail(operation, "function calls are not supported");
        return WalkResult::interrupt();
      }
      for (Region& region : operation->getRegions()) {
        if (!region.empty() && region.getBlocks().size() != 1) {
          std::ignore =
              fail(operation, "multi-block regions are not supported");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      return failure();
    }
    for (Operation& operation : moduleOp.getBody()->getOperations()) {
      if (&operation != function.getOperation()) {
        return fail(&operation, "only the entry function may appear at module "
                                "scope");
      }
    }
    return success();
  }

  [[nodiscard]] LogicalResult collectProgramShape() {
    auto returnOp =
        dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
    if (!returnOp) {
      return fail(function, "entry block must end in func.return");
    }
    for (const auto [index, value] : llvm::enumerate(returnOp.getOperands())) {
      if (returnOp.getNumOperands() == 1 && isCanonicalStatus(value, index)) {
        continue;
      }
      if (isa<cbit::RegisterType>(value.getType())) {
        returnedRegisters.insert(value);
        continue;
      }
      auto kind = inferScalarKind(value);
      if (kind.empty()) {
        return fail(returnOp, "unsupported scalar output type for function "
                              "result " +
                                  Twine(index));
      }
      scalarOutputs.push_back(
          {.value = value, .name = outputName({}), .kind = std::move(kind)});
    }

    for (Operation& operation : function.getBody().front().getOperations()) {
      if (auto alloc = dyn_cast<qc::AllocOp>(&operation)) {
        const auto name = uniqueName("q", nextQubit);
        Resource resource{.kind = ResourceKind::Qubit,
                          .name = name,
                          .width = 1,
                          .scalar = true};
        resources.try_emplace(alloc.getResult(), resource);
        resourceOrder.push_back(alloc.getResult());
        valueNames.try_emplace(alloc.getResult(), name);
        continue;
      }
      if (auto alloc = dyn_cast<cbit::AllocOp>(&operation)) {
        const auto type = alloc.getResult().getType();
        const bool isOutput = returnedRegisters.contains(alloc.getResult());
        const auto name = alloc->getAttrOfType<StringAttr>(
            mqt::MQTDialect::RegisterNameAttrHelper::getNameStr());
        const auto requested = name ? name.getValue() : StringRef{};
        Resource resource{.kind = ResourceKind::Bit,
                          .name = isOutput ? outputName(requested)
                                           : uniqueName("c", nextBit),
                          .width = type.getWidth(),
                          .output = isOutput,
                          .initialization = alloc.getInitialization()};
        resources.try_emplace(alloc.getResult(), std::move(resource));
        resourceOrder.push_back(alloc.getResult());
        continue;
      }
      auto alloc = dyn_cast<memref::AllocOp>(&operation);
      if (!alloc) {
        continue;
      }
      const auto type = dyn_cast<MemRefType>(alloc.getType());
      if (!type || !type.hasStaticShape() || type.getRank() != 1 ||
          type.getDimSize(0) <= 0) {
        return fail(alloc, "only non-empty static rank-one memrefs are "
                           "supported");
      }
      if (!isa<qc::QubitType>(type.getElementType())) {
        return fail(alloc, "only qubit memrefs are supported");
      }
      StringRef requested;
      if (const auto attr = alloc->getAttrOfType<StringAttr>(
              mqt::MQTDialect::RegisterNameAttrHelper::getNameStr())) {
        requested = attr.getValue();
      }
      Resource resource{.kind = ResourceKind::Qubit,
                        .name = qubitRegisterName(requested),
                        .width = type.getDimSize(0)};
      resources.try_emplace(alloc.getResult(), resource);
      resourceOrder.push_back(alloc.getResult());
    }

    for (auto value : returnedRegisters) {
      if (!resources.contains(value)) {
        return fail(returnOp, "returned CBit registers must be entry-block "
                              "allocations");
      }
    }
    return success();
  }

  [[nodiscard]] static bool isCanonicalStatus(Value value,
                                              const size_t resultIndex) {
    if (resultIndex != 0 || !value.getType().isInteger(64)) {
      return false;
    }
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    auto integer =
        constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
    return integer && integer.getValue().isZero();
  }

  [[nodiscard]] static std::string inferScalarKind(Value value) {
    const auto type = value.getType();
    if (type.isInteger(1)) {
      return value.getDefiningOp<qc::MeasureOp>() ? "bit" : "bool";
    }
    if (type.isInteger(64) || type.isIndex()) {
      return "int";
    }
    if (type.isF64()) {
      return "float";
    }
    return {};
  }

  [[nodiscard]] LogicalResult emitDeclarations() {
    for (auto value : resourceOrder) {
      const auto& resource = resources.at(value);
      if (resource.output) {
        *output << "output ";
      }
      *output << (resource.kind == ResourceKind::Qubit ? "qubit" : "bit");
      if (!resource.scalar) {
        *output << '[' << resource.width << ']';
      }
      *output << ' ' << resource.name << ";\n";
      if (resource.kind == ResourceKind::Bit &&
          resource.initialization == cbit::Initialization::Zero) {
        for (int64_t bit = 0; bit < resource.width; ++bit) {
          *output << resource.name << '[' << bit << "] = false;\n";
        }
      }
    }
    for (const auto& scalar : scalarOutputs) {
      *output << "output " << scalar.kind << ' ' << scalar.name << ";\n";
    }
    if (!resourceOrder.empty() || !scalarOutputs.empty()) {
      *output << '\n';
    }
    return success();
  }

  [[nodiscard]] LogicalResult emitBlock(Block& block) {
    for (Operation& operation : block.getOperations()) {
      if (isa<scf::YieldOp>(&operation)) {
        return success();
      }
      if (failed(emitOperation(operation))) {
        return failure();
      }
    }
    return success();
  }

  [[nodiscard]] LogicalResult emitOperation(Operation& operation) {
    if (isa<arith::SelectOp>(&operation)) {
      return fail(&operation, "arith.select is not supported");
    }
    if (isa<arith::ConstantOp, cbit::LoadOp, cbit::AllocOp, memref::LoadOp,
            memref::AllocOp, memref::DeallocOp, qc::AllocOp, qc::DeallocOp,
            qc::StaticOp>(&operation)) {
      return success();
    }
    if (isInlineExpressionOperation(operation)) {
      return validateInlineExpressionOperation(operation);
    }
    if (isa<cf::AssertOp>(&operation) ||
        (isa<ub::PoisonOp>(&operation) &&
         llvm::any_of(operation.getResults(),
                      [](Value result) { return !result.use_empty(); }))) {
      return fail(&operation, "runtime safety machinery is not supported");
    }
    if (isa<ub::PoisonOp>(&operation)) {
      return success();
    }
    if (auto store = dyn_cast<cbit::StoreOp>(&operation)) {
      return emitStore(store);
    }
    if (auto measurement = dyn_cast<qc::MeasureOp>(&operation)) {
      return emitMeasurement(measurement);
    }
    if (auto reset = dyn_cast<qc::ResetOp>(&operation)) {
      auto qubit = emitQubit(reset.getQubit());
      if (failed(qubit)) {
        return failure();
      }
      *output << "reset " << *qubit << ";\n";
      return success();
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(&operation)) {
      return emitIf(ifOp);
    }
    if (auto forOp = dyn_cast<scf::ForOp>(&operation)) {
      return emitFor(forOp);
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(&operation)) {
      return emitWhile(whileOp);
    }
    if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(&operation)) {
      return emitIndexSwitch(switchOp);
    }
    if (auto returnOp = dyn_cast<func::ReturnOp>(&operation)) {
      return emitReturn(returnOp);
    }
    if (auto unitary = dyn_cast<UnitaryOpInterface>(&operation)) {
      if (auto barrier = dyn_cast<BarrierOp>(&operation)) {
        SmallVector<std::string> qubits;
        for (auto value : barrier.getTargets()) {
          auto qubit = emitQubit(value);
          if (failed(qubit)) {
            return failure();
          }
          qubits.push_back(std::move(*qubit));
        }
        *output << "barrier " << llvm::join(qubits, ", ") << ";\n";
        return success();
      }
      auto call = emitGateCall(unitary);
      if (failed(call)) {
        return failure();
      }
      emitGateStatement(*call, *output);
      return success();
    }
    return fail(&operation, "unsupported operation '" +
                                operation.getName().getStringRef() + "'");
  }

  [[nodiscard]] static bool isInlineExpressionOperation(Operation& operation) {
    const auto name = operation.getName().getStringRef();
    return isa<arith::ConstantOp, arith::CmpIOp, arith::CmpFOp>(&operation) ||
           !binaryOperator(name).empty() || name == "arith.negf" ||
           name == "arith.remf" || isScalarCast(name) ||
           !mathFunction(name).empty();
  }

  [[nodiscard]] LogicalResult
  validateInlineExpressionOperation(Operation& operation) {
    for (auto result : operation.getResults()) {
      const auto type = result.getType();
      if (!type.isInteger(1) && !type.isInteger(64) && !type.isIndex() &&
          !type.isF64()) {
        return fail(&operation, "unsupported scalar expression result type");
      }
      if (failed(emitExpression(result))) {
        return failure();
      }
    }
    return success();
  }

  [[nodiscard]] FailureOr<std::string> emitQubit(Value value) {
    if (const auto found = valueNames.find(value); found != valueNames.end()) {
      return found->second;
    }
    if (auto staticOp = value.getDefiningOp<qc::StaticOp>()) {
      return (Twine('$') + Twine(staticOp.getIndex())).str();
    }
    auto load = value.getDefiningOp<memref::LoadOp>();
    if (!load || load.getIndices().size() != 1) {
      return failExpression(value, "expected a logical or physical qubit "
                                   "reference");
    }
    const auto resource = resources.find(load.getMemRef());
    if (resource == resources.end() ||
        resource->second.kind != ResourceKind::Qubit) {
      return failExpression(value, "qubit load refers to unsupported storage");
    }
    const auto index = getConstantInteger(load.getIndices().front());
    if (!index || *index < 0 || *index >= resource->second.width) {
      return failExpression(value,
                            "qubit indices must be constant and in bounds");
    }
    return (Twine(resource->second.name) + "[" + Twine(*index) + "]").str();
  }

  [[nodiscard]] FailureOr<std::string> emitBitReference(Value reg,
                                                        Value indexValue) {
    const auto resource = resources.find(reg);
    if (resource == resources.end() ||
        resource->second.kind != ResourceKind::Bit) {
      return failExpression(reg, "bit access refers to unsupported storage");
    }
    const auto index = getConstantInteger(indexValue);
    if (index && (*index < 0 || *index >= resource->second.width)) {
      return failExpression(indexValue, "constant bit index is out of bounds");
    }
    if (index) {
      return (Twine(resource->second.name) + "[" + Twine(*index) + "]").str();
    }
    auto dynamicIndex = emitExpression(indexValue);
    if (failed(dynamicIndex)) {
      return failure();
    }
    return (Twine(resource->second.name) + "[" + *dynamicIndex + "]").str();
  }

  [[nodiscard]] FailureOr<std::string> emitExpression(Value value) {
    if (const auto found = valueNames.find(value); found != valueNames.end()) {
      return found->second;
    }
    if (auto load = value.getDefiningOp<cbit::LoadOp>()) {
      return emitBitReference(load.getReg(), load.getIndex());
    }
    auto* operation = value.getDefiningOp();
    if (operation == nullptr) {
      return failExpression(value, "unmapped block argument");
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      return emitConstant(constant);
    }
    if (isa<ub::PoisonOp>(operation)) {
      return failExpression(value, "poison values are not supported");
    }
    if (auto cmp = dyn_cast<arith::CmpIOp>(operation)) {
      auto predicate = integerPredicate(cmp.getPredicate());
      if (predicate.empty()) {
        return failExpression(value, "unsupported integer comparison");
      }
      return emitBinary(cmp.getLhs(), predicate, cmp.getRhs());
    }
    if (auto cmp = dyn_cast<arith::CmpFOp>(operation)) {
      auto predicate = floatPredicate(cmp.getPredicate());
      if (predicate.empty()) {
        return failExpression(value, "unsupported floating-point comparison");
      }
      return emitBinary(cmp.getLhs(), predicate, cmp.getRhs());
    }
    const auto name = operation->getName().getStringRef();
    if (name == "arith.remf") {
      auto lhs = emitExpression(operation->getOperand(0));
      auto rhs = emitExpression(operation->getOperand(1));
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      return (Twine("mod(") + *lhs + ", " + *rhs + ")").str();
    }
    if (const auto binary = binaryOperator(name); !binary.empty()) {
      if (operation->getNumOperands() != 2) {
        return failExpression(value, "malformed binary expression");
      }
      if ((name == "arith.andi" || name == "arith.ori" ||
           name == "arith.xori") &&
          !value.getType().isInteger(1)) {
        return failExpression(value,
                              "packed integer bitwise operations are not "
                              "supported");
      }
      return emitBinary(operation->getOperand(0), binary,
                        operation->getOperand(1));
    }
    if (name == "arith.negf") {
      auto operand = emitExpression(operation->getOperand(0));
      if (failed(operand)) {
        return failure();
      }
      return (Twine("(-") + *operand + ")").str();
    }
    if (isScalarCast(name)) {
      auto operand = emitExpression(operation->getOperand(0));
      if (failed(operand)) {
        return failure();
      }
      if (name == "arith.index_cast" &&
          (operation->getOperand(0).getType().isInteger(64) ||
           operation->getOperand(0).getType().isIndex()) &&
          (value.getType().isInteger(64) || value.getType().isIndex())) {
        return operand;
      }
      auto type = castTarget(name, value.getType());
      if (type.empty()) {
        return failExpression(value, "unsupported scalar conversion");
      }
      return (Twine(type) + "(" + *operand + ")").str();
    }
    if (const auto functionName = mathFunction(name); !functionName.empty()) {
      SmallVector<std::string> arguments;
      for (auto operand : operation->getOperands()) {
        auto argument = emitExpression(operand);
        if (failed(argument)) {
          return failure();
        }
        arguments.push_back(std::move(*argument));
      }
      return (Twine(functionName) + "(" + llvm::join(arguments, ", ") + ")")
          .str();
    }
    return failExpression(value,
                          "unsupported expression operation '" + name + "'");
  }

  [[nodiscard]] static FailureOr<std::string>
  emitConstant(arith::ConstantOp constant) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue())) {
      if (integer.getType().isInteger(1)) {
        return integer.getValue().isZero() ? std::string("false")
                                           : std::string("true");
      }
      llvm::SmallString<32> text;
      integer.getValue().toString(text, 10, true);
      return text.str().str();
    }
    if (auto floating = dyn_cast<FloatAttr>(constant.getValue())) {
      const auto& value = floating.getValue();
      if (!value.isFinite()) {
        emitError(constant.getLoc())
            << "OpenQASM emission error: non-finite floating-point "
               "constants are not supported";
        return failure();
      }
      llvm::SmallString<32> text;
      value.toString(text);
      const StringRef textRef(text);
      if (!textRef.contains('.') && !textRef.contains('e') &&
          !textRef.contains('E')) {
        text.append(".0");
      }
      return text.str().str();
    }
    emitError(constant.getLoc())
        << "OpenQASM emission error: unsupported constant attribute";
    return failure();
  }

  [[nodiscard]] FailureOr<std::string>
  emitBinary(Value lhsValue, const StringRef operation, Value rhsValue) {
    auto lhs = emitExpression(lhsValue);
    auto rhs = emitExpression(rhsValue);
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }
    return (Twine("(") + *lhs + " " + operation + " " + *rhs + ")").str();
  }

  [[nodiscard]] static StringRef binaryOperator(const StringRef name) {
    return llvm::StringSwitch<StringRef>(name)
        .Cases("arith.addi", "arith.addf", "+")
        .Cases("arith.subi", "arith.subf", "-")
        .Cases("arith.muli", "arith.mulf", "*")
        .Cases("arith.divsi", "arith.divf", "/")
        .Case("arith.remsi", "%")
        .Case("arith.andi", "&&")
        .Case("arith.ori", "||")
        .Case("arith.xori", "!=")
        .Default({});
  }

  [[nodiscard]] static StringRef
  integerPredicate(const arith::CmpIPredicate predicate) {
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      return "==";
    case arith::CmpIPredicate::ne:
      return "!=";
    case arith::CmpIPredicate::slt:
      return "<";
    case arith::CmpIPredicate::sle:
      return "<=";
    case arith::CmpIPredicate::sgt:
      return ">";
    case arith::CmpIPredicate::sge:
      return ">=";
    case arith::CmpIPredicate::ult:
    case arith::CmpIPredicate::ule:
    case arith::CmpIPredicate::ugt:
    case arith::CmpIPredicate::uge:
      return {};
    }
    return {};
  }

  [[nodiscard]] static StringRef
  floatPredicate(const arith::CmpFPredicate predicate) {
    switch (predicate) {
    case arith::CmpFPredicate::OEQ:
      return "==";
    case arith::CmpFPredicate::ONE:
      return "!=";
    case arith::CmpFPredicate::OLT:
      return "<";
    case arith::CmpFPredicate::OLE:
      return "<=";
    case arith::CmpFPredicate::OGT:
      return ">";
    case arith::CmpFPredicate::OGE:
      return ">=";
    default:
      return {};
    }
  }

  [[nodiscard]] static bool isScalarCast(const StringRef name) {
    return llvm::StringSwitch<bool>(name)
        .Case("arith.index_cast", true)
        .Case("arith.sitofp", true)
        .Case("arith.fptosi", true)
        .Default(false);
  }

  [[nodiscard]] static StringRef castTarget(const StringRef name,
                                            const Type resultType) {
    if (name == "arith.sitofp" || resultType.isF64()) {
      return "float";
    }
    if (resultType.isInteger(1)) {
      return "bool";
    }
    if (resultType.isInteger(64) || resultType.isIndex()) {
      return "int";
    }
    return {};
  }

  [[nodiscard]] static StringRef mathFunction(const StringRef name) {
    return llvm::StringSwitch<StringRef>(name)
        .Case("math.acos", "arccos")
        .Case("math.asin", "arcsin")
        .Case("math.atan", "arctan")
        .Case("math.ceil", "ceiling")
        .Case("math.cos", "cos")
        .Case("math.exp", "exp")
        .Case("math.floor", "floor")
        .Case("math.log", "log")
        .Case("math.powf", "pow")
        .Case("math.sin", "sin")
        .Case("math.sqrt", "sqrt")
        .Case("math.tan", "tan")
        .Default({});
  }

  [[nodiscard]] LogicalResult emitStore(cbit::StoreOp store) {
    auto target = emitBitReference(store.getReg(), store.getIndex());
    if (failed(target)) {
      return failure();
    }
    if (auto measurement = store.getValue().getDefiningOp<qc::MeasureOp>();
        measurement && measurement.getResult().hasOneUse() &&
        measurement->getNextNode() == store.getOperation()) {
      auto qubit = emitQubit(measurement.getQubit());
      if (failed(qubit)) {
        return failure();
      }
      *output << *target << " = measure " << *qubit << ";\n";
      return success();
    }
    auto value = emitExpression(store.getValue());
    if (failed(value)) {
      return failure();
    }
    *output << *target << " = " << *value << ";\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitMeasurement(qc::MeasureOp measurement) {
    if (measurement.getResult().hasOneUse()) {
      if (auto store = dyn_cast<cbit::StoreOp>(
              *measurement.getResult().getUsers().begin());
          store && measurement->getNextNode() == store.getOperation()) {
        return success();
      }
    }
    auto qubit = emitQubit(measurement.getQubit());
    if (failed(qubit)) {
      return failure();
    }
    const auto name = uniqueName("b", nextBit);
    valueNames.try_emplace(measurement.getResult(), name);
    *output << "bit " << name << " = measure " << *qubit << ";\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitIf(scf::IfOp ifOp) {
    if (ifOp.getNumResults() != 0) {
      return fail(ifOp, "scf.if results are not supported");
    }
    auto condition = emitExpression(ifOp.getCondition());
    if (failed(condition)) {
      return failure();
    }
    *output << "if (" << *condition << ") {\n";
    output->indent();
    if (failed(emitBlock(ifOp.getThenRegion().front()))) {
      return failure();
    }
    output->unindent();
    if (!ifOp.getElseRegion().empty()) {
      *output << "} else {\n";
      output->indent();
      if (failed(emitBlock(ifOp.getElseRegion().front()))) {
        return failure();
      }
      output->unindent();
    }
    *output << "}\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitFor(scf::ForOp forOp) {
    if (!forOp.getInitArgs().empty() || forOp.getNumResults() != 0) {
      return fail(forOp, "scf.for loop-carried values are not supported");
    }
    const auto lower = getConstantInteger(forOp.getLowerBound());
    const auto upper = getConstantInteger(forOp.getUpperBound());
    const auto step = getConstantInteger(forOp.getStep());
    if (!lower || !upper || !step || *step <= 0) {
      return fail(forOp, "scf.for requires constant bounds and a positive "
                         "constant step");
    }
    if (*lower >= *upper) {
      return success();
    }
    const APInt lowerWide(65, static_cast<uint64_t>(*lower), true);
    const APInt upperWide(65, static_cast<uint64_t>(*upper), true);
    const APInt stepWide(65, static_cast<uint64_t>(*step), true);
    const auto lastWide =
        lowerWide + ((upperWide - 1 - lowerWide).sdiv(stepWide) * stepWide);
    const auto last = lastWide.getSExtValue();

    const auto induction = uniqueName("i", nextLoop);
    valueNames.try_emplace(forOp.getInductionVar(), induction);

    *output << "for int " << induction << " in [" << *lower;
    if (*step != 1) {
      *output << ':' << *step;
    }
    *output << ':' << last << "] {\n";
    output->indent();
    if (failed(emitBlock(*forOp.getBody()))) {
      return failure();
    }
    output->unindent();
    *output << "}\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitWhile(scf::WhileOp whileOp) {
    auto& before = whileOp.getBefore().front();
    auto& after = whileOp.getAfter().front();
    auto conditionOp = cast<scf::ConditionOp>(before.getTerminator());
    auto yieldOp = cast<scf::YieldOp>(after.getTerminator());
    if (!whileOp.getInits().empty() || whileOp.getNumResults() != 0 ||
        before.getNumArguments() != 0 || after.getNumArguments() != 0 ||
        !conditionOp.getArgs().empty() || yieldOp.getNumOperands() != 0) {
      return fail(whileOp, "scf.while loop-carried values are not supported");
    }
    for (Operation& operation : before.without_terminator()) {
      if (auto load = dyn_cast<cbit::LoadOp>(operation)) {
        if (failed(emitExpression(load.getResult()))) {
          return failure();
        }
        continue;
      }
      if (!isInlineExpressionOperation(operation) ||
          !isMemoryEffectFree(&operation)) {
        return fail(&operation,
                    "scf.while condition region must be side-effect free");
      }
      if (failed(validateInlineExpressionOperation(operation))) {
        return failure();
      }
    }
    auto condition = emitExpression(conditionOp.getCondition());
    if (failed(condition)) {
      return failure();
    }
    *output << "while (" << *condition << ") {\n";
    output->indent();
    if (failed(emitBlock(after))) {
      return failure();
    }
    output->unindent();
    *output << "}\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitIndexSwitch(scf::IndexSwitchOp switchOp) {
    if (switchOp.getNumResults() != 0) {
      return fail(switchOp, "scf.index_switch results are not supported");
    }
    auto argument = emitExpression(switchOp.getArg());
    if (failed(argument)) {
      return failure();
    }

    const auto cases = switchOp.getCases();
    if (cases.empty()) {
      return emitBlock(switchOp.getDefaultBlock());
    }
    *output << "switch (" << *argument << ") {\n";
    output->indent();
    for (const auto [index, caseValue] : llvm::enumerate(cases)) {
      *output << "case " << caseValue << " {\n";
      output->indent();
      if (failed(
              emitBlock(switchOp.getCaseBlock(static_cast<unsigned>(index))))) {
        return failure();
      }
      output->unindent();
      *output << "}\n";
    }
    *output << "default {\n";
    output->indent();
    if (failed(emitBlock(switchOp.getDefaultBlock()))) {
      return failure();
    }
    output->unindent();
    *output << "}\n";
    output->unindent();
    *output << "}\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitReturn(func::ReturnOp returnOp) {
    size_t scalarIndex = 0;
    for (const auto [index, value] : llvm::enumerate(returnOp.getOperands())) {
      if (returnOp.getNumOperands() == 1 && isCanonicalStatus(value, index)) {
        continue;
      }
      if (isa<cbit::RegisterType>(value.getType())) {
        continue;
      }
      auto expression = emitExpression(value);
      if (failed(expression)) {
        return failure();
      }
      *output << scalarOutputs[scalarIndex].name << " = " << *expression
              << ";\n";
      ++scalarIndex;
    }
    return success();
  }

  [[nodiscard]] FailureOr<GateCall> emitGateCall(UnitaryOpInterface unitary) {
    if (auto ctrl = dyn_cast<CtrlOp>(unitary.getOperation())) {
      return emitModifier(ctrl);
    }
    if (auto inverse = dyn_cast<InvOp>(unitary.getOperation())) {
      return emitModifier(inverse);
    }
    if (auto power = dyn_cast<PowOp>(unitary.getOperation())) {
      return emitModifier(power);
    }

    GateCall call;
    const auto baseSymbol = unitary.getBaseSymbol();
    auto symbol = portableGateSymbol(baseSymbol);
    if (failed(symbol)) {
      return failure();
    }
    call.symbol = std::move(*symbol);
    if (baseSymbol == "sxdg") {
      call.modifiers = "inv @ ";
    }
    for (auto parameter : unitary.getParameters()) {
      auto expression = emitExpression(parameter);
      if (failed(expression)) {
        return failure();
      }
      call.parameters.push_back(std::move(*expression));
    }
    for (auto qubitValue : unitary.getTargets()) {
      auto qubit = emitQubit(qubitValue);
      if (failed(qubit)) {
        return failure();
      }
      call.qubits.push_back(std::move(*qubit));
    }
    return call;
  }

  template <typename ModifierOp>
  [[nodiscard]] FailureOr<GateCall> emitModifier(ModifierOp modifier) {
    auto& body = modifier.getRegion().front();
    SmallVector<Operation*> unitaries;
    for (Operation& operation : body.without_terminator()) {
      if (!isa<UnitaryOpInterface>(&operation) &&
          !isInlineExpressionOperation(operation)) {
        fail(&operation, "modifier bodies may only contain unitary operations "
                         "and scalar expressions");
        return failure();
      }
      if (isa<UnitaryOpInterface>(&operation)) {
        unitaries.push_back(&operation);
      }
    }
    if (unitaries.empty()) {
      fail(modifier, "modifier body contains no unitary operation");
      return failure();
    }

    SmallVector<std::string> targets;
    for (auto target : modifier.getTargets()) {
      auto qubit = emitQubit(target);
      if (failed(qubit)) {
        return failure();
      }
      targets.push_back(std::move(*qubit));
    }
    if (body.getNumArguments() != targets.size()) {
      fail(modifier, "modifier target and body argument counts differ");
      return failure();
    }
    for (const auto [argument, target] :
         llvm::zip_equal(body.getArguments(), targets)) {
      valueNames.try_emplace(argument, target);
    }

    GateCall call;
    if (unitaries.size() == 1) {
      auto nested = emitGateCall(cast<UnitaryOpInterface>(unitaries.front()));
      if (failed(nested)) {
        return failure();
      }
      call = std::move(*nested);
    } else {
      auto helper = createCompositeHelper(modifier, unitaries);
      if (failed(helper)) {
        return failure();
      }
      call = std::move(*helper);
      call.qubits = targets;
    }

    if constexpr (std::is_same_v<ModifierOp, CtrlOp>) {
      SmallVector<std::string> controls;
      for (auto control : modifier.getControls()) {
        auto qubit = emitQubit(control);
        if (failed(qubit)) {
          return failure();
        }
        controls.push_back(std::move(*qubit));
      }
      call.qubits.insert(call.qubits.begin(), controls.begin(), controls.end());
      call.modifiers =
          (Twine("ctrl") +
           (controls.size() == 1 ? Twine{}
                                 : Twine("(") + Twine(controls.size()) + ")") +
           " @ " + call.modifiers)
              .str();
    } else if constexpr (std::is_same_v<ModifierOp, InvOp>) {
      call.modifiers = (Twine("inv @ ") + call.modifiers).str();
    } else {
      auto exponent = emitExpression(modifier.getExponent());
      if (failed(exponent)) {
        return failure();
      }
      call.modifiers =
          (Twine("pow(") + *exponent + ") @ " + call.modifiers).str();
    }
    return call;
  }

  template <typename ModifierOp>
  [[nodiscard]] FailureOr<GateCall>
  createCompositeHelper(ModifierOp modifier,
                        const ArrayRef<Operation*> unitaries) {
    auto& body = modifier.getRegion().front();
    if (body.getNumArguments() == 0) {
      fail(modifier, "multi-operation modifiers require a target qubit");
      return failure();
    }
    const auto helperName = uniqueName("gate", nextHelper);

    SmallVector<Value> captures;
    DenseSet<Value> captured;
    Value capturedQubit;
    modifier.getRegion().walk([&](Operation* operation) {
      for (auto operand : operation->getOperands()) {
        if (modifier.getRegion().isAncestor(operand.getParentRegion())) {
          continue;
        }
        if (isa<QubitType>(operand.getType())) {
          capturedQubit = operand;
        } else if (captured.insert(operand).second) {
          captures.push_back(operand);
        }
      }
    });
    if (capturedQubit) {
      fail(modifier,
           "multi-operation modifier bodies cannot capture extra qubits");
      return failure();
    }

    GateCall helperCall;
    helperCall.symbol = helperName;
    for (auto capture : captures) {
      auto expression = emitExpression(capture);
      if (failed(expression)) {
        return failure();
      }
      helperCall.parameters.push_back(std::move(*expression));
    }

    DenseMap<Value, std::string> savedNames;
    auto saveAndMap = [&](Value value, std::string name) {
      if (const auto found = valueNames.find(value);
          found != valueNames.end()) {
        savedNames.try_emplace(value, found->second);
      }
      valueNames[value] = std::move(name);
    };

    SmallVector<std::string> parameterNames;
    for (const auto [index, value] : llvm::enumerate(captures)) {
      auto name = (Twine("p") + Twine(index)).str();
      parameterNames.push_back(name);
      saveAndMap(value, std::move(name));
    }
    SmallVector<std::string> qubitNames;
    for (const auto [index, argument] : llvm::enumerate(body.getArguments())) {
      auto name = (Twine("q") + Twine(index)).str();
      qubitNames.push_back(name);
      saveAndMap(argument, std::move(name));
    }

    std::string definition;
    llvm::raw_string_ostream definitionStream(definition);
    raw_indented_ostream definitionOutput(definitionStream);
    definitionOutput << "gate " << helperName;
    if (!parameterNames.empty()) {
      definitionOutput << '(' << llvm::join(parameterNames, ", ") << ')';
    }
    definitionOutput << ' ' << llvm::join(qubitNames, ", ") << " {\n";
    definitionOutput.indent();
    auto* savedOutput = output;
    output = &definitionOutput;
    for (auto* operation : unitaries) {
      auto call = emitGateCall(cast<UnitaryOpInterface>(operation));
      if (failed(call)) {
        output = savedOutput;
        return failure();
      }
      emitGateStatement(*call, definitionOutput);
    }
    output = savedOutput;
    definitionOutput.unindent();
    definitionOutput << "}\n";
    definitionOutput.flush();
    compositeHelpers.push_back(std::move(definition));

    for (auto value : captures) {
      if (const auto found = savedNames.find(value);
          found != savedNames.end()) {
        valueNames[value] = found->second;
      } else {
        valueNames.erase(value);
      }
    }
    for (auto argument : body.getArguments()) {
      if (const auto found = savedNames.find(argument);
          found != savedNames.end()) {
        valueNames[argument] = found->second;
      } else {
        valueNames.erase(argument);
      }
    }
    return helperCall;
  }

  static void emitGateStatement(const GateCall& call,
                                raw_indented_ostream& stream) {
    stream << call.modifiers << call.symbol;
    if (!call.parameters.empty()) {
      stream << '(' << llvm::join(call.parameters, ", ") << ')';
    }
    if (!call.qubits.empty()) {
      stream << ' ' << llvm::join(call.qubits, ", ");
    }
    stream << ";\n";
  }

  [[nodiscard]] FailureOr<std::string>
  portableGateSymbol(const StringRef symbol) {
    if (symbol == "sxdg") {
      return std::string("sx");
    }
    if (symbol == "u") {
      return std::string("U");
    }
    const auto* gate = oq3::frontend::lookupGate(symbol);
    if (gate == nullptr ||
        gate->availability == oq3::frontend::GateAvailability::QELib1) {
      emitError(function.getLoc())
          << "OpenQASM emission error: unsupported quantum gate '" << symbol
          << "'";
      return failure();
    }
    if (gate->availability == oq3::frontend::GateAvailability::Compatibility) {
      fixedHelpers.insert(symbol);
    }
    return symbol.str();
  }

  void emitFixedHelpers(llvm::raw_ostream& stream) const {
    using HelperDefinition = std::pair<StringLiteral, StringLiteral>;
    constexpr std::array helpers{
        HelperDefinition{"r", "gate r(p0, p1) q {\n"
                              "  rz(-p1) q;\n"
                              "  rx(p0) q;\n"
                              "  rz(p1) q;\n"
                              "}\n"},
        HelperDefinition{"iswap", "gate iswap q0, q1 {\n"
                                  "  s q0;\n"
                                  "  s q1;\n"
                                  "  h q0;\n"
                                  "  ctrl @ x q0, q1;\n"
                                  "  ctrl @ x q1, q0;\n"
                                  "  h q1;\n"
                                  "}\n"},
        HelperDefinition{"dcx", "gate dcx q0, q1 {\n"
                                "  ctrl @ x q0, q1;\n"
                                "  ctrl @ x q1, q0;\n"
                                "}\n"},
        HelperDefinition{"rxx", "gate rxx(p0) q0, q1 {\n"
                                "  h q0;\n"
                                "  h q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "  rz(p0) q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "  h q0;\n"
                                "  h q1;\n"
                                "}\n"},
        HelperDefinition{"ryy", "gate ryy(p0) q0, q1 {\n"
                                "  rx(pi / 2) q0;\n"
                                "  rx(pi / 2) q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "  rz(p0) q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "  rx(-pi / 2) q0;\n"
                                "  rx(-pi / 2) q1;\n"
                                "}\n"},
        HelperDefinition{"rzz", "gate rzz(p0) q0, q1 {\n"
                                "  ctrl @ x q0, q1;\n"
                                "  rz(p0) q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "}\n"},
        HelperDefinition{"rzx", "gate rzx(p0) q0, q1 {\n"
                                "  h q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "  rz(p0) q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "  h q1;\n"
                                "}\n"},
        HelperDefinition{"ecr", "gate ecr q0, q1 {\n"
                                "  gphase(-pi / 4);\n"
                                "  s q0;\n"
                                "  sx q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "  x q0;\n"
                                "}\n"},
        HelperDefinition{"rccx", "gate rccx q0, q1, q2 {\n"
                                 "  h q2;\n"
                                 "  t q2;\n"
                                 "  ctrl @ x q1, q2;\n"
                                 "  tdg q2;\n"
                                 "  ctrl @ x q0, q2;\n"
                                 "  t q2;\n"
                                 "  ctrl @ x q1, q2;\n"
                                 "  tdg q2;\n"
                                 "  h q2;\n"
                                 "}\n"},
        HelperDefinition{"xx_plus_yy", "gate xx_plus_yy(p0, p1) q0, q1 {\n"
                                       "  rz(p1) q0;\n"
                                       "  sdg q1;\n"
                                       "  s q0;\n"
                                       "  sx q1;\n"
                                       "  s q1;\n"
                                       "  ctrl @ x q1, q0;\n"
                                       "  ry(-p0 / 2) q0;\n"
                                       "  ry(-p0 / 2) q1;\n"
                                       "  ctrl @ x q1, q0;\n"
                                       "  sdg q0;\n"
                                       "  sdg q1;\n"
                                       "  rz(-p1) q0;\n"
                                       "  inv @ sx q1;\n"
                                       "  s q1;\n"
                                       "}\n"},
        HelperDefinition{"xx_minus_yy", "gate xx_minus_yy(p0, p1) q0, q1 {\n"
                                        "  sdg q0;\n"
                                        "  rz(-p1) q1;\n"
                                        "  sx q0;\n"
                                        "  s q1;\n"
                                        "  s q0;\n"
                                        "  ctrl @ x q0, q1;\n"
                                        "  ry(p0 / 2) q0;\n"
                                        "  ry(-p0 / 2) q1;\n"
                                        "  ctrl @ x q0, q1;\n"
                                        "  sdg q0;\n"
                                        "  sdg q1;\n"
                                        "  inv @ sx q0;\n"
                                        "  rz(p1) q1;\n"
                                        "  s q0;\n"
                                        "}\n"},
    };
    for (const auto& helper : helpers) {
      if (fixedHelpers.contains(helper.first)) {
        stream << helper.second << '\n';
      }
    }
  }
};

} // namespace

LogicalResult translateQCToOpenQASM3(ModuleOp moduleOp,
                                     llvm::raw_ostream& output) {
  auto source = translateQCToOpenQASM3(moduleOp);
  if (failed(source)) {
    return failure();
  }
  output << *source;
  return success();
}

FailureOr<std::string> translateQCToOpenQASM3(ModuleOp moduleOp) {
  return OpenQASMEmitter(moduleOp).emit();
}

} // namespace mlir::qc
