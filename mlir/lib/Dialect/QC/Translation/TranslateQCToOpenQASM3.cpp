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

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/IndentedOstream.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
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

[[nodiscard]] bool isOpenQASMIdentifier(const StringRef value) {
  if (value.empty() ||
      !(llvm::isAlpha(value.front()) || value.front() == '_')) {
    return false;
  }
  return llvm::all_of(value.drop_front(), [](const char character) {
    return llvm::isAlnum(character) || character == '_';
  });
}

[[nodiscard]] bool isReservedOpenQASMIdentifier(const StringRef value) {
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

[[nodiscard]] bool isValidOutputName(const StringRef value) {
  return isOpenQASMIdentifier(value) && !value.starts_with("_mqt_") &&
         !isReservedOpenQASMIdentifier(value);
}

[[nodiscard]] std::optional<int64_t> getConstantInteger(const Value value) {
  return getConstantIntValue(value);
}

[[nodiscard]] std::string join(const ArrayRef<std::string> values,
                               const StringRef separator) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  llvm::interleave(values, stream, separator);
  return result;
}

class OpenQASMEmitter {
public:
  explicit OpenQASMEmitter(const ModuleOp moduleOp) : moduleOp(moduleOp) {}

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
        failed(emitBlock(function.getBody().front(), {}, {}, false))) {
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
  DenseSet<Value> returnedMemrefs;
  SmallVector<ScalarOutput> scalarOutputs;
  llvm::StringSet<> usedNames;
  llvm::StringSet<> fixedHelpers;
  SmallVector<std::string> compositeHelpers;
  size_t nextQubit = 0;
  size_t nextBit = 0;
  size_t nextScalar = 0;
  size_t nextLoop = 0;
  size_t nextHelper = 0;

  [[nodiscard]] LogicalResult fail(Operation* operation,
                                   const Twine& message) const {
    operation->emitError() << "OpenQASM 3 emission error: " << message;
    return failure();
  }

  [[nodiscard]] FailureOr<std::string>
  failExpression(const Value value, const Twine& message) const {
    emitError(value.getLoc()) << "OpenQASM 3 emission error: " << message;
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

  [[nodiscard]] static StringAttr getResultStringAttr(func::FuncOp funcOp,
                                                      const unsigned index,
                                                      const StringRef name) {
    if (const auto attrs = funcOp.getResultAttrDict(index)) {
      return attrs.getAs<StringAttr>(name);
    }
    return {};
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
        fail(operation, "function calls are not supported");
        return WalkResult::interrupt();
      }
      for (Region& region : operation->getRegions()) {
        if (!region.empty() && region.getBlocks().size() != 1) {
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
    if (returnOp.getNumOperands() != function.getNumResults()) {
      return fail(returnOp, "function result and return operand counts differ");
    }

    for (const auto [index, value] : llvm::enumerate(returnOp.getOperands())) {
      if (returnOp.getNumOperands() == 1 && isCanonicalStatus(value, index)) {
        continue;
      }
      const auto nameAttr = getResultStringAttr(
          function, static_cast<unsigned>(index), OPENQASM_OUTPUT_NAME_ATTR);
      const auto kindAttr = getResultStringAttr(
          function, static_cast<unsigned>(index), OPENQASM_OUTPUT_KIND_ATTR);
      if (isa<MemRefType>(value.getType())) {
        returnedMemrefs.insert(value);
        continue;
      }
      auto kind = kindAttr ? kindAttr.getValue().str() : inferScalarKind(value);
      if (kind.empty()) {
        return fail(returnOp,
                    "cannot infer signedness or an OpenQASM type for function "
                    "result " +
                        Twine(index));
      }
      if (!isCompatibleScalarKind(kind, value.getType())) {
        return fail(returnOp, "OpenQASM output kind '" + Twine(kind) +
                                  "' conflicts with function result " +
                                  Twine(index));
      }
      scalarOutputs.push_back(
          {.value = value,
           .name = outputName(nameAttr ? nameAttr.getValue() : StringRef{}),
           .kind = std::move(kind)});
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
      Resource resource{.kind = isa<qc::QubitType>(type.getElementType())
                                    ? ResourceKind::Qubit
                                    : ResourceKind::Bit,
                        .width = type.getDimSize(0)};
      if (resource.kind == ResourceKind::Bit &&
          !type.getElementType().isInteger(1)) {
        return fail(alloc, "only qubit and i1 memrefs are supported");
      }
      if (resource.kind == ResourceKind::Qubit) {
        resource.name = uniqueName("q", nextQubit);
      } else {
        resource.output = returnedMemrefs.contains(alloc.getResult());
        StringRef requested;
        size_t returnIndex = returnOp.getNumOperands();
        if (resource.output) {
          returnIndex = llvm::find(returnOp.getOperands(), alloc.getResult()) -
                        returnOp.getOperands().begin();
          if (returnIndex < returnOp.getNumOperands()) {
            if (const auto nameAttr = getResultStringAttr(
                    function, static_cast<unsigned>(returnIndex),
                    OPENQASM_OUTPUT_NAME_ATTR)) {
              requested = nameAttr.getValue();
            }
          }
        }
        if (const auto attr = alloc->getAttrOfType<StringAttr>(
                utils::CLASSICAL_REGISTER_NAME_ATTR);
            requested.empty() && attr) {
          requested = attr.getValue();
        }
        resource.name =
            resource.output ? outputName(requested) : uniqueName("c", nextBit);
        if (returnIndex < returnOp.getNumOperands()) {
          if (const auto kindAttr = getResultStringAttr(
                  function, static_cast<unsigned>(returnIndex),
                  OPENQASM_OUTPUT_KIND_ATTR)) {
            if (kindAttr.getValue() != "bit" &&
                kindAttr.getValue() != "bit_array") {
              return fail(returnOp, "bit memref output kind must be 'bit' or "
                                    "'bit_array'");
            }
            resource.scalar = kindAttr.getValue() == "bit";
          }
        }
      }
      resources.try_emplace(alloc.getResult(), resource);
      resourceOrder.push_back(alloc.getResult());
    }

    for (const auto value : returnedMemrefs) {
      if (!resources.contains(value)) {
        return fail(returnOp,
                    "returned memrefs must be entry-block allocations");
      }
      auto& resource = resources.at(value);
      if (resource.kind != ResourceKind::Bit) {
        return fail(returnOp, "only classical bit memrefs may be outputs");
      }
      if (resource.scalar && resource.width != 1) {
        return fail(returnOp, "a scalar bit output must have width one");
      }
    }
    return success();
  }

  [[nodiscard]] bool isCanonicalStatus(const Value value,
                                       const size_t resultIndex) const {
    if (resultIndex != 0 || !value.getType().isInteger(64)) {
      return false;
    }
    if (getResultStringAttr(function, static_cast<unsigned>(resultIndex),
                            OPENQASM_OUTPUT_NAME_ATTR) ||
        getResultStringAttr(function, static_cast<unsigned>(resultIndex),
                            OPENQASM_OUTPUT_KIND_ATTR)) {
      return false;
    }
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    auto integer =
        constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
    return integer && integer.getValue().isZero();
  }

  [[nodiscard]] static std::string inferScalarKind(const Value value) {
    return TypeSwitch<Type, std::string>(value.getType())
        .Case<IntegerType>([&](const IntegerType type) {
          if (type.getWidth() == 1) {
            if (value.getDefiningOp<qc::MeasureOp>()) {
              return std::string("bit");
            }
            if (auto load = value.getDefiningOp<memref::LoadOp>()) {
              if (auto memref = dyn_cast<MemRefType>(load.getMemRefType());
                  memref && memref.getElementType().isInteger(1)) {
                return std::string("bit");
              }
            }
            return std::string("bool");
          }
          if (type.getWidth() != 64) {
            return std::string{};
          }
          auto* operation = value.getDefiningOp();
          if (operation == nullptr) {
            return std::string{};
          }
          const auto name = operation->getName().getStringRef();
          if (name == "arith.divui" || name == "arith.remui" ||
              name == "arith.extui" || name == "arith.index_castui" ||
              name == "arith.fptoui") {
            return std::string("uint");
          }
          if (name == "arith.divsi" || name == "arith.remsi" ||
              name == "arith.extsi" || name == "arith.fptosi") {
            return std::string("int");
          }
          return std::string{};
        })
        .Case<FloatType>([](FloatType type) {
          return type.getWidth() == 64 ? std::string("float") : std::string{};
        })
        .Default([](const Type) { return std::string{}; });
  }

  [[nodiscard]] static bool isCompatibleScalarKind(const StringRef kind,
                                                   const Type type) {
    if (type.isInteger(1)) {
      return kind == "bit" || kind == "bool";
    }
    if (type.isInteger(64)) {
      return kind == "int" || kind == "uint";
    }
    if (type.isF64()) {
      return kind == "float";
    }
    return false;
  }

  [[nodiscard]] LogicalResult emitDeclarations() {
    for (const auto value : resourceOrder) {
      const auto& resource = resources.at(value);
      if (resource.output) {
        *output << "output ";
      }
      *output << (resource.kind == ResourceKind::Qubit ? "qubit" : "bit");
      if (!resource.scalar) {
        *output << '[' << resource.width << ']';
      }
      *output << ' ' << resource.name << ";\n";
    }
    for (const auto& scalar : scalarOutputs) {
      *output << "output " << scalar.kind << ' ' << scalar.name << ";\n";
    }
    if (!resourceOrder.empty() || !scalarOutputs.empty()) {
      *output << '\n';
    }
    return success();
  }

  [[nodiscard]] FailureOr<std::string> emitType(const Type type,
                                                const StringRef hint = {}) {
    if (!hint.empty()) {
      return hint.str();
    }
    if (type.isInteger(1)) {
      return std::string("bool");
    }
    if (type.isInteger(64) || type.isIndex()) {
      return std::string("int");
    }
    if (type.isF64()) {
      return std::string("float");
    }
    emitError(function.getLoc())
        << "OpenQASM 3 emission error: unsupported scalar type " << type;
    return failure();
  }

  [[nodiscard]] LogicalResult
  emitBlock(Block& block, const ArrayRef<std::string> yieldTargets,
            const ArrayRef<Type> yieldTypes, const bool simultaneousYield) {
    for (Operation& operation : block.getOperations()) {
      if (auto yield = dyn_cast<scf::YieldOp>(&operation)) {
        return emitYield(yield, yieldTargets, yieldTypes, simultaneousYield);
      }
      if (isa<scf::ConditionOp, qc::YieldOp>(&operation)) {
        return fail(&operation, "unexpected region terminator");
      }
      if (failed(emitOperation(operation))) {
        return failure();
      }
    }
    if (!yieldTargets.empty()) {
      return fail(block.getParentOp(), "structured region has no scf.yield");
    }
    return success();
  }

  [[nodiscard]] LogicalResult emitOperation(Operation& operation) {
    if (auto select = dyn_cast<arith::SelectOp>(&operation)) {
      return emitSelect(select);
    }
    if (isa<arith::ConstantOp, memref::LoadOp, memref::AllocOp,
            memref::DeallocOp, qc::AllocOp, qc::DeallocOp, qc::StaticOp>(
            &operation)) {
      return success();
    }
    if (isInlineExpressionOperation(operation)) {
      return validateInlineExpressionOperation(operation);
    }
    if (isa<cf::AssertOp>(&operation) ||
        (isa<ub::PoisonOp>(&operation) &&
         llvm::any_of(operation.getResults(), [](const Value result) {
           return !result.use_empty();
         }))) {
      return fail(&operation, "runtime safety machinery is not supported");
    }
    if (isa<ub::PoisonOp>(&operation)) {
      return success();
    }
    if (auto store = dyn_cast<memref::StoreOp>(&operation)) {
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
        for (const auto value : barrier.getTargets()) {
          auto qubit = emitQubit(value);
          if (failed(qubit)) {
            return failure();
          }
          qubits.push_back(std::move(*qubit));
        }
        *output << "barrier " << join(qubits, ", ") << ";\n";
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
           isScalarCast(name) || !mathFunction(name).empty();
  }

  [[nodiscard]] LogicalResult
  validateInlineExpressionOperation(Operation& operation) {
    DenseSet<Operation*> visited;
    if (isDeadExpressionTree(operation, visited)) {
      return success();
    }
    if (operation.getNumResults() == 0) {
      return fail(&operation, "malformed scalar expression operation");
    }
    for (const auto result : operation.getResults()) {
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

  [[nodiscard]] static bool
  isDeadExpressionTree(Operation& operation, DenseSet<Operation*>& visited) {
    if (!visited.insert(&operation).second) {
      return true;
    }
    for (const auto result : operation.getResults()) {
      for (Operation* user : result.getUsers()) {
        const auto name = user->getName().getStringRef();
        if ((!name.starts_with("arith.") && !name.starts_with("math.")) ||
            !isMemoryEffectFree(user) ||
            !isDeadExpressionTree(*user, visited)) {
          return false;
        }
      }
    }
    return true;
  }

  [[nodiscard]] FailureOr<std::string> emitQubit(const Value value) {
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

  [[nodiscard]] FailureOr<std::string>
  emitBitReference(const Value memref, const Value indexValue) {
    const auto resource = resources.find(memref);
    if (resource == resources.end() ||
        resource->second.kind != ResourceKind::Bit) {
      return failExpression(memref, "bit access refers to unsupported storage");
    }
    const auto index = getConstantInteger(indexValue);
    if (!index || *index < 0 || *index >= resource->second.width) {
      return failExpression(indexValue,
                            "bit indices must be constant and in bounds");
    }
    if (resource->second.scalar) {
      if (*index != 0) {
        return failExpression(indexValue,
                              "scalar bit output may only use index zero");
      }
      return resource->second.name;
    }
    return (Twine(resource->second.name) + "[" + Twine(*index) + "]").str();
  }

  [[nodiscard]] FailureOr<std::string> emitExpression(const Value value) {
    if (const auto found = valueNames.find(value); found != valueNames.end()) {
      return found->second;
    }
    if (auto load = value.getDefiningOp<memref::LoadOp>()) {
      if (load.getIndices().size() != 1) {
        return failExpression(value, "only rank-one loads are supported");
      }
      return emitBitReference(load.getMemRef(), load.getIndices().front());
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
    if (auto select = dyn_cast<arith::SelectOp>(operation)) {
      if (const auto found = valueNames.find(select.getResult());
          found != valueNames.end()) {
        return found->second;
      }
      return failExpression(value,
                            "arith.select must be materialized before use");
    }

    const auto name = operation->getName().getStringRef();
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
      for (const auto operand : operation->getOperands()) {
        auto argument = emitExpression(operand);
        if (failed(argument)) {
          return failure();
        }
        arguments.push_back(std::move(*argument));
      }
      return (Twine(functionName) + "(" + join(arguments, ", ") + ")").str();
    }
    return failExpression(value,
                          "unsupported expression operation '" + name + "'");
  }

  [[nodiscard]] FailureOr<std::string>
  emitConstant(arith::ConstantOp constant) const {
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
            << "OpenQASM 3 emission error: non-finite floating-point "
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
        << "OpenQASM 3 emission error: unsupported constant attribute";
    return failure();
  }

  [[nodiscard]] FailureOr<std::string> emitBinary(const Value lhsValue,
                                                  const StringRef operation,
                                                  const Value rhsValue) {
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
        .Cases("arith.divsi", "arith.divui", "arith.divf", "/")
        .Cases("arith.remsi", "arith.remui", "arith.remf", "%")
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
    case arith::CmpIPredicate::ult:
      return "<";
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return "<=";
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return ">";
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return ">=";
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
        .Cases("arith.extsi", "arith.extui", "arith.trunci", true)
        .Cases("arith.index_cast", "arith.index_castui", true)
        .Cases("arith.sitofp", "arith.uitofp", true)
        .Cases("arith.fptosi", "arith.fptoui", true)
        .Default(false);
  }

  [[nodiscard]] static StringRef castTarget(const StringRef name,
                                            const Type resultType) {
    if (name == "arith.sitofp" || name == "arith.uitofp" ||
        resultType.isF64()) {
      return "float";
    }
    if (name == "arith.fptoui" || name == "arith.extui" ||
        name == "arith.index_castui") {
      return "uint";
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

  [[nodiscard]] LogicalResult emitStore(memref::StoreOp store) {
    if (store.getIndices().size() != 1) {
      return fail(store, "only rank-one stores are supported");
    }
    auto target =
        emitBitReference(store.getMemRef(), store.getIndices().front());
    auto value = emitExpression(store.getValue());
    if (failed(target) || failed(value)) {
      return failure();
    }
    *output << *target << " = " << *value << ";\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitMeasurement(qc::MeasureOp measurement) {
    auto qubit = emitQubit(measurement.getQubit());
    if (failed(qubit)) {
      return failure();
    }
    const auto name = uniqueName("b", nextBit);
    valueNames.try_emplace(measurement.getResult(), name);
    *output << "bit " << name << " = measure " << *qubit << ";\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitSelect(arith::SelectOp select) {
    auto type = emitType(select.getType());
    auto condition = emitExpression(select.getCondition());
    auto trueValue = emitExpression(select.getTrueValue());
    auto falseValue = emitExpression(select.getFalseValue());
    if (failed(type) || failed(condition) || failed(trueValue) ||
        failed(falseValue)) {
      return failure();
    }
    const auto name = uniqueName("v", nextScalar);
    valueNames.try_emplace(select.getResult(), name);
    *output << *type << ' ' << name << ";\n";
    *output << "if (" << *condition << ") {\n";
    output->indent();
    *output << name << " = " << *trueValue << ";\n";
    output->unindent();
    *output << "} else {\n";
    output->indent();
    *output << name << " = " << *falseValue << ";\n";
    output->unindent();
    *output << "}\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitIf(scf::IfOp ifOp) {
    auto condition = emitExpression(ifOp.getCondition());
    if (failed(condition)) {
      return failure();
    }
    SmallVector<std::string> resultNames;
    SmallVector<Type> resultTypes;
    for (const auto result : ifOp.getResults()) {
      auto type = emitType(result.getType());
      if (failed(type)) {
        return failure();
      }
      const auto name = uniqueName("v", nextScalar);
      valueNames.try_emplace(result, name);
      resultNames.push_back(name);
      resultTypes.push_back(result.getType());
      *output << *type << ' ' << name << ";\n";
    }
    *output << "if (" << *condition << ") {\n";
    output->indent();
    if (failed(emitBlock(ifOp.getThenRegion().front(), resultNames, resultTypes,
                         false))) {
      return failure();
    }
    output->unindent();
    if (!ifOp.getElseRegion().empty()) {
      *output << "} else {\n";
      output->indent();
      if (failed(emitBlock(ifOp.getElseRegion().front(), resultNames,
                           resultTypes, false))) {
        return failure();
      }
      output->unindent();
    }
    *output << "}\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitFor(scf::ForOp forOp) {
    const auto lower = getConstantInteger(forOp.getLowerBound());
    const auto upper = getConstantInteger(forOp.getUpperBound());
    const auto step = getConstantInteger(forOp.getStep());
    if (!lower || !upper || !step || *step <= 0) {
      return fail(forOp, "scf.for requires constant bounds and a positive "
                         "constant step");
    }
    if (*lower >= *upper) {
      for (const auto [result, initial] :
           llvm::zip_equal(forOp.getResults(), forOp.getInitArgs())) {
        auto expression = emitExpression(initial);
        if (failed(expression)) {
          return failure();
        }
        valueNames.try_emplace(result, std::move(*expression));
      }
      return success();
    }
    const APInt lowerWide(65, static_cast<uint64_t>(*lower), true);
    const APInt upperWide(65, static_cast<uint64_t>(*upper), true);
    const APInt stepWide(65, static_cast<uint64_t>(*step), true);
    const auto lastWide =
        lowerWide + ((upperWide - 1 - lowerWide).sdiv(stepWide) * stepWide);
    const auto last = lastWide.getSExtValue();

    SmallVector<std::string> stateNames;
    SmallVector<Type> stateTypes;
    for (const auto [result, initial] :
         llvm::zip_equal(forOp.getResults(), forOp.getInitArgs())) {
      auto type = emitType(result.getType());
      auto expression = emitExpression(initial);
      if (failed(type) || failed(expression)) {
        return failure();
      }
      const auto name = uniqueName("state", nextLoop);
      valueNames.try_emplace(result, name);
      stateNames.push_back(name);
      stateTypes.push_back(result.getType());
      *output << *type << ' ' << name << " = " << *expression << ";\n";
    }

    const auto induction = uniqueName("i", nextLoop);
    valueNames.try_emplace(forOp.getInductionVar(), induction);
    for (const auto [argument, name] :
         llvm::zip_equal(forOp.getRegionIterArgs(), stateNames)) {
      valueNames.try_emplace(argument, name);
    }

    *output << "for int " << induction << " in [" << *lower;
    if (*step != 1) {
      *output << ':' << *step;
    }
    *output << ':' << last << "] {\n";
    output->indent();
    if (failed(emitBlock(*forOp.getBody(), stateNames, stateTypes, true))) {
      return failure();
    }
    output->unindent();
    *output << "}\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitWhile(scf::WhileOp whileOp) {
    auto& before = whileOp.getBefore().front();
    auto& after = whileOp.getAfter().front();
    auto conditionOp = dyn_cast<scf::ConditionOp>(before.getTerminator());
    auto yieldOp = dyn_cast<scf::YieldOp>(after.getTerminator());
    if (!conditionOp || !yieldOp ||
        conditionOp.getArgs().size() != after.getNumArguments() ||
        whileOp.getInits().size() != before.getNumArguments()) {
      return fail(whileOp, "malformed scf.while regions");
    }
    for (const auto [initial, result, beforeArgument, afterArgument] :
         llvm::zip_equal(whileOp.getInits(), whileOp.getResults(),
                         before.getArguments(), after.getArguments())) {
      if (initial.getType() != result.getType() ||
          beforeArgument.getType() != result.getType() ||
          afterArgument.getType() != result.getType()) {
        return fail(whileOp,
                    "scf.while requires type-preserving carried state");
      }
    }

    SmallVector<std::string> stateNames;
    SmallVector<Type> stateTypes;
    for (const auto [result, initial] :
         llvm::zip_equal(whileOp.getResults(), whileOp.getInits())) {
      auto type = emitType(result.getType());
      auto expression = emitExpression(initial);
      if (failed(type) || failed(expression)) {
        return failure();
      }
      const auto name = uniqueName("state", nextLoop);
      valueNames.try_emplace(result, name);
      stateNames.push_back(name);
      stateTypes.push_back(result.getType());
      *output << *type << ' ' << name << " = " << *expression << ";\n";
    }
    for (const auto [argument, name] :
         llvm::zip_equal(before.getArguments(), stateNames)) {
      valueNames.try_emplace(argument, name);
    }
    for (const auto [forwarded, beforeArgument, afterArgument] :
         llvm::zip_equal(conditionOp.getArgs(), before.getArguments(),
                         after.getArguments())) {
      if (forwarded != beforeArgument) {
        return fail(conditionOp,
                    "scf.while condition may only forward carried values "
                    "unchanged");
      }
      valueNames.try_emplace(afterArgument, valueNames.at(beforeArgument));
    }
    for (Operation& operation : before.without_terminator()) {
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
    if (failed(emitBlock(after, stateNames, stateTypes, true))) {
      return failure();
    }
    output->unindent();
    *output << "}\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitIndexSwitch(scf::IndexSwitchOp switchOp) {
    auto argument = emitExpression(switchOp.getArg());
    if (failed(argument)) {
      return failure();
    }

    SmallVector<std::string> resultNames;
    SmallVector<Type> resultTypes;
    for (const auto result : switchOp.getResults()) {
      auto type = emitType(result.getType());
      if (failed(type)) {
        return failure();
      }
      const auto name = uniqueName("v", nextScalar);
      valueNames.try_emplace(result, name);
      resultNames.push_back(name);
      resultTypes.push_back(result.getType());
      *output << *type << ' ' << name << ";\n";
    }

    const auto cases = switchOp.getCases();
    if (cases.empty()) {
      return emitBlock(switchOp.getDefaultBlock(), resultNames, resultTypes,
                       false);
    }
    for (const auto [index, caseValue] : llvm::enumerate(cases)) {
      if (index != 0) {
        *output << "} else {\n";
        output->indent();
      }
      *output << "if (" << *argument << " == " << caseValue << ") {\n";
      output->indent();
      if (failed(emitBlock(switchOp.getCaseBlock(static_cast<unsigned>(index)),
                           resultNames, resultTypes, false))) {
        return failure();
      }
      output->unindent();
    }
    *output << "} else {\n";
    output->indent();
    if (failed(emitBlock(switchOp.getDefaultBlock(), resultNames, resultTypes,
                         false))) {
      return failure();
    }
    output->unindent();
    *output << "}\n";
    for (size_t index = 1; index < cases.size(); ++index) {
      output->unindent();
      *output << "}\n";
    }
    return success();
  }

  [[nodiscard]] LogicalResult emitYield(scf::YieldOp yield,
                                        const ArrayRef<std::string> targets,
                                        const ArrayRef<Type> types,
                                        const bool simultaneous) {
    if (yield.getNumOperands() != targets.size() ||
        targets.size() != types.size()) {
      return fail(yield, "scf.yield arity does not match region results");
    }
    SmallVector<std::string> expressions;
    for (const auto operand : yield.getOperands()) {
      auto expression = emitExpression(operand);
      if (failed(expression)) {
        return failure();
      }
      expressions.push_back(std::move(*expression));
    }
    if (!simultaneous) {
      for (const auto [target, expression] :
           llvm::zip_equal(targets, expressions)) {
        *output << target << " = " << expression << ";\n";
      }
      return success();
    }
    SmallVector<std::string> temporaries;
    for (const auto [typeValue, expression] :
         llvm::zip_equal(types, expressions)) {
      auto type = emitType(typeValue);
      if (failed(type)) {
        return failure();
      }
      const auto temporary = uniqueName("next", nextLoop);
      temporaries.push_back(temporary);
      *output << *type << ' ' << temporary << " = " << expression << ";\n";
    }
    for (const auto [target, temporary] :
         llvm::zip_equal(targets, temporaries)) {
      *output << target << " = " << temporary << ";\n";
    }
    return success();
  }

  [[nodiscard]] LogicalResult emitReturn(func::ReturnOp returnOp) {
    size_t scalarIndex = 0;
    for (const auto [index, value] : llvm::enumerate(returnOp.getOperands())) {
      if (returnOp.getNumOperands() == 1 && isCanonicalStatus(value, index)) {
        continue;
      }
      if (isa<MemRefType>(value.getType())) {
        continue;
      }
      if (scalarIndex >= scalarOutputs.size()) {
        return fail(returnOp, "missing scalar output declaration");
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
    auto symbol = portableGateSymbol(unitary.getBaseSymbol());
    if (failed(symbol)) {
      return failure();
    }
    call.symbol = std::move(*symbol);
    for (const auto parameter : unitary.getParameters()) {
      auto expression = emitExpression(parameter);
      if (failed(expression)) {
        return failure();
      }
      call.parameters.push_back(std::move(*expression));
    }
    for (const auto qubitValue : unitary.getTargets()) {
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
    for (const auto target : modifier.getTargets()) {
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
      for (const auto control : modifier.getControls()) {
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
    const auto helperName = uniqueName("gate", nextHelper);

    SmallVector<Value> dependencies;
    for (const auto* operation : unitaries) {
      auto unitary = cast<UnitaryOpInterface>(operation);
      llvm::append_range(dependencies, unitary.getParameters());
    }
    SmallVector<Value> captures;
    DenseSet<Value> visited;
    for (size_t index = 0; index < dependencies.size(); ++index) {
      auto dependency = dependencies[index];
      if (!visited.insert(dependency).second) {
        continue;
      }
      if (dependency.getParentRegion() != &modifier.getRegion()) {
        captures.push_back(dependency);
        continue;
      }
      if (auto* definingOperation = dependency.getDefiningOp()) {
        for (const auto operand : definingOperation->getOperands()) {
          if (!isa<QubitType>(operand.getType())) {
            dependencies.push_back(operand);
          }
        }
      }
    }

    GateCall helperCall;
    helperCall.symbol = helperName;
    for (const auto capture : captures) {
      auto expression = emitExpression(capture);
      if (failed(expression)) {
        return failure();
      }
      helperCall.parameters.push_back(std::move(*expression));
    }

    DenseMap<Value, std::string> savedNames;
    auto saveAndMap = [&](const Value value, std::string name) {
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
      definitionOutput << '(' << join(parameterNames, ", ") << ')';
    }
    definitionOutput << ' ' << join(qubitNames, ", ") << " {\n";
    definitionOutput.indent();
    auto* savedOutput = output;
    output = &definitionOutput;
    for (const auto* operation : unitaries) {
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

    for (const auto value : captures) {
      if (const auto found = savedNames.find(value);
          found != savedNames.end()) {
        valueNames[value] = found->second;
      } else {
        valueNames.erase(value);
      }
    }
    for (const auto argument : body.getArguments()) {
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
      stream << '(' << join(call.parameters, ", ") << ')';
    }
    if (!call.qubits.empty()) {
      stream << ' ' << join(call.qubits, ", ");
    }
    stream << ";\n";
  }

  [[nodiscard]] FailureOr<std::string>
  portableGateSymbol(const StringRef symbol) {
    constexpr std::array STANDARD_GATES{
        StringLiteral("gphase"), StringLiteral("id"),  StringLiteral("x"),
        StringLiteral("y"),      StringLiteral("z"),   StringLiteral("h"),
        StringLiteral("s"),      StringLiteral("sdg"), StringLiteral("t"),
        StringLiteral("tdg"),    StringLiteral("sx"),  StringLiteral("p"),
        StringLiteral("rx"),     StringLiteral("ry"),  StringLiteral("rz"),
        StringLiteral("swap"),
    };
    if (llvm::is_contained(STANDARD_GATES, symbol)) {
      return symbol.str();
    }
    constexpr std::array HELPER_GATES{
        StringLiteral("sxdg"),        StringLiteral("r"),
        StringLiteral("u2"),          StringLiteral("u"),
        StringLiteral("iswap"),       StringLiteral("dcx"),
        StringLiteral("ecr"),         StringLiteral("rxx"),
        StringLiteral("ryy"),         StringLiteral("rzx"),
        StringLiteral("rzz"),         StringLiteral("xx_plus_yy"),
        StringLiteral("xx_minus_yy"), StringLiteral("rccx"),
    };
    if (!llvm::is_contained(HELPER_GATES, symbol)) {
      emitError(function.getLoc())
          << "OpenQASM 3 emission error: unsupported quantum gate '" << symbol
          << "'";
      return failure();
    }
    fixedHelpers.insert(symbol);
    return (Twine("_mqt_") + symbol).str();
  }

  void emitFixedHelpers(llvm::raw_ostream& stream) const {
    struct HelperDefinition {
      StringLiteral symbol;
      StringLiteral definition;
    };
    constexpr std::array HELPERS{
        HelperDefinition{"sxdg", "gate _mqt_sxdg q {\n  inv @ sx q;\n}\n"},
        HelperDefinition{"r", "gate _mqt_r(p0, p1) q {\n"
                              "  rz(-p1) q;\n"
                              "  rx(p0) q;\n"
                              "  rz(p1) q;\n"
                              "}\n"},
        HelperDefinition{"u2", "gate _mqt_u2(p0, p1) q {\n"
                               "  U(pi / 2, p0, p1) q;\n"
                               "}\n"},
        HelperDefinition{"u", "gate _mqt_u(p0, p1, p2) q {\n"
                              "  U(p0, p1, p2) q;\n"
                              "}\n"},
        HelperDefinition{"iswap", "gate _mqt_iswap q0, q1 {\n"
                                  "  s q0;\n"
                                  "  s q1;\n"
                                  "  h q0;\n"
                                  "  ctrl @ x q0, q1;\n"
                                  "  ctrl @ x q1, q0;\n"
                                  "  h q1;\n"
                                  "}\n"},
        HelperDefinition{"dcx", "gate _mqt_dcx q0, q1 {\n"
                                "  ctrl @ x q0, q1;\n"
                                "  ctrl @ x q1, q0;\n"
                                "}\n"},
        HelperDefinition{"rxx", "gate _mqt_rxx(p0) q0, q1 {\n"
                                "  h q0;\n"
                                "  h q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "  rz(p0) q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "  h q0;\n"
                                "  h q1;\n"
                                "}\n"},
        HelperDefinition{"ryy", "gate _mqt_ryy(p0) q0, q1 {\n"
                                "  rx(pi / 2) q0;\n"
                                "  rx(pi / 2) q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "  rz(p0) q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "  rx(-pi / 2) q0;\n"
                                "  rx(-pi / 2) q1;\n"
                                "}\n"},
        HelperDefinition{"rzz", "gate _mqt_rzz(p0) q0, q1 {\n"
                                "  ctrl @ x q0, q1;\n"
                                "  rz(p0) q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "}\n"},
        HelperDefinition{"rzx", "gate _mqt_rzx(p0) q0, q1 {\n"
                                "  h q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "  rz(p0) q1;\n"
                                "  ctrl @ x q0, q1;\n"
                                "  h q1;\n"
                                "}\n"},
        HelperDefinition{"ecr", "gate _mqt_ecr q0, q1 {\n"
                                "  _mqt_rzx(pi / 4) q0, q1;\n"
                                "  x q0;\n"
                                "  _mqt_rzx(-pi / 4) q0, q1;\n"
                                "}\n"},
        HelperDefinition{"rccx", "gate _mqt_rccx q0, q1, q2 {\n"
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
        HelperDefinition{"xx_plus_yy", "gate _mqt_xx_plus_yy(p0, p1) q0, q1 {\n"
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
        HelperDefinition{"xx_minus_yy",
                         "gate _mqt_xx_minus_yy(p0, p1) q0, q1 {\n"
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
    for (const auto& helper : HELPERS) {
      if (fixedHelpers.contains(helper.symbol)) {
        if (helper.symbol == "ecr" && !fixedHelpers.contains("rzx")) {
          const auto rzx = llvm::find_if(HELPERS, [](const auto& candidate) {
            return candidate.symbol == "rzx";
          });
          stream << rzx->definition << '\n';
        }
        stream << helper.definition << '\n';
      }
    }
  }
};

} // namespace

LogicalResult translateQCToOpenQASM3(const ModuleOp moduleOp,
                                     llvm::raw_ostream& output) {
  auto source = translateQCToOpenQASM3(moduleOp);
  if (failed(source)) {
    return failure();
  }
  output << *source;
  return success();
}

FailureOr<std::string> translateQCToOpenQASM3(const ModuleOp moduleOp) {
  return OpenQASMEmitter(moduleOp).emit();
}

} // namespace mlir::qc
