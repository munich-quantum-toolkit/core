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
#include "mlir/Support/IntegerExpressions.h"
#include "mlir/Target/OpenQASM/GateCatalog.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/SaveAndRestore.h>
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
      .Cases({"OPENQASM", "include", "input", "output", "const"}, true)
      .Cases({"let", "fixed", "gate", "def", "extern"}, true)
      .Cases({"defcalgrammar", "defcal", "cal", "opaque", "box"}, true)
      .Cases({"delay", "reset", "measure", "barrier"}, true)
      .Cases({"ctrl", "negctrl", "inv", "pow"}, true)
      .Cases({"if", "else", "while", "for", "in"}, true)
      .Cases({"break", "continue", "end", "return"}, true)
      .Cases({"switch", "case", "default"}, true)
      .Cases({"qubit", "qreg", "creg", "bit", "bool"}, true)
      .Cases({"int", "uint", "float", "angle", "complex"}, true)
      .Cases({"array", "duration", "stretch", "readonly", "mutable"}, true)
      .Cases({"sizeof", "durationof", "true", "false"}, true)
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

    indexClassicalWrites(function.getBody().front());
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
    sourceStream << measurementDeclarations << body;
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
  std::string measurementDeclarations;
  DenseMap<Operation*, size_t> operationPositions;
  DenseMap<Block*, DenseMap<Value, SmallVector<size_t>>> classicalWrites;
  Operation* expressionConsumer = nullptr;
  size_t nextQubit = 0;
  size_t nextBit = 0;
  size_t nextScalar = 0;
  size_t nextLoop = 0;
  size_t nextHelper = 0;
  bool materializeScalars = false;
  size_t expressionNesting = 0;
  size_t expressionWork = 0;
  size_t numClassicalBits = 0;

  static constexpr size_t MAX_EXPRESSION_NESTING = 256;
  static constexpr size_t MAX_EXPRESSION_WORK = 4096;
  static constexpr size_t MAX_CLASSICAL_BITS = 1U << 20;

  [[nodiscard]] static LogicalResult fail(Operation* operation,
                                          const Twine& message) {
    operation->emitError() << "OpenQASM emission error: " << message
                           << " (operation " << operation->getName() << ")";
    return failure();
  }

  [[nodiscard]] static FailureOr<std::string>
  failExpression(Value value, const Twine& message) {
    emitError(value.getLoc()) << "OpenQASM emission error: " << message
                              << " (value type " << value.getType() << ")";
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
        Resource resource{
            .kind = ResourceKind::Qubit,
            .name = name,
            .width = 1,
            .scalar = true,
        };
        resources.try_emplace(alloc.getResult(), resource);
        resourceOrder.push_back(alloc.getResult());
        valueNames.try_emplace(alloc.getResult(), name);
        continue;
      }
      if (auto alloc = dyn_cast<cbit::AllocOp>(&operation)) {
        const auto type = alloc.getResult().getType();
        const auto width = type.getWidth();
        if (width <= 0 ||
            std::cmp_greater(width, MAX_CLASSICAL_BITS - numClassicalBits)) {
          return fail(alloc, "total classical register width exceeds the "
                             "supported limit of " +
                                 Twine(MAX_CLASSICAL_BITS) + " bits");
        }
        numClassicalBits += static_cast<size_t>(width);
        const bool isOutput = returnedRegisters.contains(alloc.getResult());
        const auto name = alloc->getAttrOfType<StringAttr>(
            mqt::MQTDialect::RegisterNameAttrHelper::getNameStr());
        const auto requested = name ? name.getValue() : StringRef{};
        Resource resource{
            .kind = ResourceKind::Bit,
            .name = isOutput ? outputName(requested) : uniqueName("c", nextBit),
            .width = width,
            .output = isOutput,
            .initialization = alloc.getInitialization(),
        };
        resources.try_emplace(alloc.getResult(), std::move(resource));
        resourceOrder.push_back(alloc.getResult());
        continue;
      }
      auto alloc = dyn_cast<memref::AllocOp>(&operation);
      if (!alloc) {
        continue;
      }
      const auto type = alloc.getType();
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
      Resource resource{
          .kind = ResourceKind::Qubit,
          .name = qubitRegisterName(requested),
          .width = type.getDimSize(0),
      };
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
    if (auto integer = dyn_cast<IntegerType>(type);
        integer && integer.getWidth() <= 64) {
      return (Twine("uint[") + Twine(integer.getWidth()) + "]").str();
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
      if (isa<scf::YieldOp, scf::ConditionOp>(&operation)) {
        return success();
      }
      if (failed(emitOperation(operation))) {
        return failure();
      }
    }
    return success();
  }

  [[nodiscard]] LogicalResult emitOperation(Operation& operation) {
    llvm::SaveAndRestore consumerGuard(expressionConsumer, &operation);
    if (materializeScalars && isInlineExpressionOperation(operation) &&
        !isa<arith::ConstantOp>(&operation) && operation.getNumResults() == 1 &&
        !(isa<cbit::ReadOp>(operation) &&
          cast<IntegerType>(operation.getResult(0).getType()).getWidth() >
              64U) &&
        !operation.getResult(0).use_empty()) {
      return materialize(operation.getResult(0));
    }
    if (isa<qc::AllocOp, memref::AllocOp, cbit::AllocOp>(&operation) &&
        operation.getBlock() != &function.getBody().front()) {
      return fail(&operation, "resource allocation inside control flow is not "
                              "supported; allocate resources before the loop");
    }
    if (isa<arith::ConstantOp, cbit::LoadOp, cbit::ReadOp, cbit::AllocOp,
            memref::LoadOp, memref::AllocOp, memref::DeallocOp, qc::AllocOp,
            qc::DeallocOp, qc::StaticOp>(&operation)) {
      return success();
    }
    if (isInlineExpressionOperation(operation)) {
      return success();
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
    if (auto write = dyn_cast<cbit::WriteOp>(&operation)) {
      const auto resource = resources.find(write.getReg());
      if (resource == resources.end() ||
          resource->second.kind != ResourceKind::Bit) {
        return fail(write, "register write refers to unsupported storage");
      }
      auto value =
          emitExpression(write.getValue(), ExpressionContext::BitVector);
      if (succeeded(value) && resource->second.width <= 64) {
        value = (Twine("bit[") + Twine(resource->second.width) + "](uint[" +
                 Twine(resource->second.width) + "](" + *value + "))")
                    .str();
      }
      if (failed(value)) {
        return failure();
      }
      *output << resource->second.name;
      *output << " = " << *value << ";\n";
      return success();
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
    return isa<arith::ConstantOp, arith::CmpIOp, arith::CmpFOp, cbit::LoadOp,
               cbit::ReadOp, arith::SelectOp, arith::ExtSIOp, arith::ExtUIOp,
               arith::TruncIOp, arith::ShRSIOp>(&operation) ||
           !binaryOperator(name).empty() || name == "arith.negf" ||
           name == "arith.remf" || name == "llvm.intr.fshl" ||
           name == "llvm.intr.fshr" || isScalarCast(name) ||
           !mathFunction(name).empty();
  }

  void indexClassicalWrites(Block& block) {
    size_t position = 0;
    for (Operation& operation : block) {
      operationPositions[&operation] = position;
      DenseSet<Value> writtenRegisters;
      operation.walk([&](Operation* nested) {
        if (auto store = dyn_cast<cbit::StoreOp>(nested)) {
          writtenRegisters.insert(store.getReg());
        } else if (auto write = dyn_cast<cbit::WriteOp>(nested)) {
          writtenRegisters.insert(write.getReg());
        }
      });
      for (auto reg : writtenRegisters) {
        classicalWrites[&block][reg].push_back(position);
      }
      for (Region& region : operation.getRegions()) {
        for (Block& nested : region) {
          indexClassicalWrites(nested);
        }
      }
      ++position;
    }
  }

  [[nodiscard]] LogicalResult validateClassicalSnapshot(Operation* read,
                                                        Value reg) {
    if (expressionConsumer == nullptr ||
        read->getBlock() != expressionConsumer->getBlock()) {
      return fail(read, "cannot preserve a classical snapshot across a "
                        "control-flow region");
    }
    const auto readPosition = operationPositions.at(read);
    const auto consumerPosition = operationPositions.at(expressionConsumer);
    if (readPosition > consumerPosition) {
      return fail(read, "classical snapshot does not dominate its use");
    }
    const auto blockWrites = classicalWrites.find(read->getBlock());
    if (blockWrites == classicalWrites.end()) {
      return success();
    }
    const auto registerWrites = blockWrites->second.find(reg);
    if (registerWrites == blockWrites->second.end()) {
      return success();
    }
    auto* const nextWrite =
        std::upper_bound(registerWrites->second.begin(),
                         registerWrites->second.end(), readPosition);
    if (nextWrite != registerWrites->second.end() &&
        *nextWrite < consumerPosition) {
      return fail(read, "cannot preserve a stale classical snapshot");
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

  enum class ExpressionContext : uint8_t { Scalar, BitVector };

  [[nodiscard]] FailureOr<std::string>
  emitExpression(Value value,
                 const ExpressionContext context = ExpressionContext::Scalar) {
    if (expressionNesting == 0) {
      expressionWork = 0;
    }
    llvm::SaveAndRestore depthGuard(expressionNesting, expressionNesting + 1);
    ++expressionWork;
    if (expressionNesting > MAX_EXPRESSION_NESTING) {
      return failExpression(value, "expression nesting exceeds the supported "
                                   "maximum of " +
                                       Twine(MAX_EXPRESSION_NESTING));
    }
    if (expressionWork > MAX_EXPRESSION_WORK) {
      return failExpression(value, "expression expansion exceeds the supported "
                                   "maximum of " +
                                       Twine(MAX_EXPRESSION_WORK) + " values");
    }
    const auto type = value.getType();
    if (const auto found = valueNames.find(value); found != valueNames.end()) {
      return found->second;
    }
    if (auto load = value.getDefiningOp<cbit::LoadOp>()) {
      if (failed(validateClassicalSnapshot(load, load.getReg()))) {
        return failure();
      }
      return emitBitReference(load.getReg(), load.getIndex());
    }
    if (auto read = value.getDefiningOp<cbit::ReadOp>()) {
      if (failed(validateClassicalSnapshot(read, read.getReg()))) {
        return failure();
      }
      const auto resource = resources.find(read.getReg());
      if (resource == resources.end() ||
          resource->second.kind != ResourceKind::Bit) {
        return failExpression(value,
                              "register read refers to unsupported storage");
      }
      return context == ExpressionContext::Scalar && type.isInteger(1)
                 ? resource->second.name + "[0]"
                 : resource->second.name;
    }
    auto* operation = value.getDefiningOp();
    if (operation == nullptr) {
      return failExpression(value, "unmapped block argument");
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      return emitConstant(constant, context != ExpressionContext::Scalar);
    }
    if (isa<ub::PoisonOp>(operation)) {
      return failExpression(value, "poison values are not supported");
    }
    const auto integer = [&](Value operand,
                             bool isSigned = false) -> FailureOr<std::string> {
      auto text = emitExpression(operand, ExpressionContext::BitVector);
      if (failed(text)) {
        return failure();
      }
      const auto operandType = dyn_cast<IntegerType>(operand.getType());
      if (!operandType) {
        return text;
      }
      const auto width = operandType.getWidth();
      if (width > 64) {
        return isSigned
                   ? failExpression(operand,
                                    "signed integers support at most 64 bits")
                   : text;
      }
      return (Twine(isSigned ? "int[" : "uint[") + Twine(width) + "](" + *text +
              ")")
          .str();
    };
    if (auto cmp = dyn_cast<arith::CmpIOp>(operation)) {
      const bool isSigned =
          mqt::unsignedPredicate(cmp.getPredicate()) != cmp.getPredicate();
      auto lhs = integer(cmp.getLhs(), isSigned);
      auto rhs = integer(cmp.getRhs(), isSigned);
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      return (Twine("(") + *lhs + " " + integerPredicate(cmp.getPredicate()) +
              " " + *rhs + ")")
          .str();
    }
    if (auto selection = dyn_cast<arith::SelectOp>(operation)) {
      auto condition = emitExpression(selection.getCondition());
      auto lhs = integer(selection.getTrueValue());
      auto rhs = integer(selection.getFalseValue());
      if (failed(condition) || failed(lhs) || failed(rhs)) {
        return failure();
      }
      const auto integerType = dyn_cast<IntegerType>(type);
      if (!integerType || integerType.getWidth() > 64) {
        return failExpression(
            value, "selection requires an integer of at most 64 bits");
      }
      const auto width = Twine(integerType.getWidth()).str();
      const auto mask =
          "uint[" + width + "](0 - uint[" + width + "](" + *condition + "))";
      const auto selected =
          "(" + *rhs + " ^ ((" + *lhs + " ^ " + *rhs + ") & " + mask + "))";
      return type.isInteger(1) ? "bool(" + selected + ")" : selected;
    }
    if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp>(operation)) {
      auto input = operation->getOperand(0);
      if (cast<IntegerType>(type).getWidth() > 64 ||
          (cast<IntegerType>(input.getType()).getWidth() > 64 &&
           (input.getDefiningOp() == nullptr ||
            input.getDefiningOp()->getName().getStringRef() != "math.ctpop"))) {
        return failExpression(value, "integer casts support at most 64 bits");
      }
      auto operand = integer(input, isa<arith::ExtSIOp>(operation));
      if (failed(operand)) {
        return failure();
      }
      const auto width = cast<IntegerType>(type).getWidth();
      if (width == 1) {
        return (Twine("((") + *operand + " & uint[" +
                Twine(std::min(cast<IntegerType>(input.getType()).getWidth(),
                               64U)) +
                "](1)) != 0)")
            .str();
      }
      return (Twine("uint[") + Twine(width) + "](" + *operand + ")").str();
    }
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivUIOp,
            arith::DivSIOp, arith::RemUIOp, arith::RemSIOp, arith::AndIOp,
            arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRUIOp,
            arith::ShRSIOp>(operation) &&
        isa<IntegerType>(type)) {
      const auto width = cast<IntegerType>(type).getWidth();
      if (width > 64 &&
          !isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(operation)) {
        return failExpression(value,
                              "integer arithmetic supports at most 64 bits");
      }
      const bool isSigned =
          isa<arith::DivSIOp, arith::RemSIOp, arith::ShRSIOp>(operation);
      auto lhs = integer(operation->getOperand(0), isSigned);
      auto rhs = integer(operation->getOperand(1),
                         isSigned && !isa<arith::ShRSIOp>(operation));
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(operation)) {
        /// Unsigned machine arithmetic preserves every narrower modular result.
        *lhs = "uint[64](" + *lhs + ")";
        *rhs = "uint[64](" + *rhs + ")";
      }
      if (isa<arith::ShRSIOp>(operation)) {
        /// OpenQASM only has a zero-filling shift. Bias the sign bit around it.
        auto source = integer(operation->getOperand(0));
        if (failed(source)) {
          return failExpression(value,
                                "signed right shifts support at most 64 bits");
        }
        const auto sign = (Twine("uint[") + Twine(width) + "](" +
                           Twine(uint64_t{1} << (width - 1)) + ")")
                              .str();
        return (Twine("uint[") + Twine(width) + "](uint[64](((" + *source +
                " ^ " + sign + ") >> " + *rhs + ")) - uint[64](" + sign +
                " >> " + *rhs + "))")
            .str();
      }
      const auto name = operation->getName().getStringRef();
      const auto op = isa<arith::AndIOp>(operation)   ? StringRef("&")
                      : isa<arith::OrIOp>(operation)  ? StringRef("|")
                      : isa<arith::XOrIOp>(operation) ? StringRef("^")
                                                      : binaryOperator(name);
      const auto expression =
          (Twine("(") + *lhs + " " + op + " " + *rhs + ")").str();
      if (width == 1) {
        return (Twine("(uint[1](") + expression + ") != 0)").str();
      }
      return width > 64
                 ? expression
                 : (Twine("uint[") + Twine(width) + "](" + expression + ")")
                       .str();
    }
    if (auto cmp = dyn_cast<arith::CmpFOp>(operation)) {
      auto predicate = floatPredicate(cmp.getPredicate());
      if (predicate.empty()) {
        return failExpression(value, "unsupported floating-point comparison");
      }
      return emitBinary(cmp.getLhs(), predicate, cmp.getRhs());
    }
    const auto name = operation->getName().getStringRef();
    if (name == "llvm.intr.fshl" || name == "llvm.intr.fshr") {
      if (operation->getNumOperands() != 3 ||
          operation->getOperand(0) != operation->getOperand(1)) {
        return failExpression(
            value, "only rotations with identical data operands are supported");
      }
      auto operand = emitExpression(operation->getOperand(0),
                                    ExpressionContext::BitVector);
      const auto width = cast<IntegerType>(type).getWidth();
      auto count = operation->getOperand(2);
      if (auto extension = count.getDefiningOp<arith::ExtUIOp>()) {
        count = extension.getIn();
      }
      FailureOr<std::string> distance = failure();
      if (auto constant = count.getDefiningOp<arith::ConstantOp>()) {
        const auto bits = cast<IntegerAttr>(constant.getValue()).getValue();
        distance = std::to_string(bits.urem(width));
      } else if (cast<IntegerType>(count.getType()).getWidth() <= 64) {
        distance = integer(count);
        if (succeeded(distance)) {
          /// OpenQASM rotations take signed counts; reduce before interpreting.
          *distance = (Twine("int[64](uint[64](") + *distance +
                       ") % uint[64](" + Twine(width) + "))")
                          .str();
        }
      }
      if (failed(operand) || failed(distance)) {
        return failExpression(value, "rotation counts support at most 64 bits");
      }
      if (width <= 64) {
        *operand = (Twine("bit[") + Twine(width) + "](uint[" + Twine(width) +
                    "](" + *operand + "))")
                       .str();
      }
      auto rotation = (Twine(name == "llvm.intr.fshl" ? "rotl(" : "rotr(") +
                       *operand + ", " + *distance + ")")
                          .str();
      return width > 64
                 ? rotation
                 : (Twine("uint[") + Twine(width) + "](" + rotation + ")")
                       .str();
    }
    if (name == "arith.remf") {
      auto lhs = emitExpression(operation->getOperand(0));
      if (failed(lhs)) {
        return failure();
      }
      auto rhs = emitExpression(operation->getOperand(1));
      if (failed(rhs)) {
        return failure();
      }
      return (Twine("mod(") + *lhs + ", " + *rhs + ")").str();
    }
    if (const auto binary = binaryOperator(name); !binary.empty()) {
      if (operation->getNumOperands() != 2) {
        return failExpression(value, "malformed binary expression");
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
      auto input = operation->getOperand(0);
      auto operand = isa<IntegerType>(input.getType())
                         ? integer(input, name != "arith.uitofp")
                         : emitExpression(input);
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
    if (name == "math.ctpop") {
      auto operand = integer(operation->getOperand(0));
      if (failed(operand)) {
        return failure();
      }
      const auto width = cast<IntegerType>(type).getWidth();
      if (width > 64) {
        return (Twine("popcount(") + *operand + ")").str();
      }
      return (Twine("uint[") + Twine(width) + "](popcount(bit[" + Twine(width) +
              "](" + *operand + ")))")
          .str();
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
  emitConstant(arith::ConstantOp constant,
               const bool bitVectorContext = false) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue())) {
      if (integer.getType().isInteger(1) && !bitVectorContext) {
        return integer.getValue().isZero() ? std::string("false")
                                           : std::string("true");
      }
      llvm::SmallString<32> text;
      integer.getValue().toString(text, 10, !bitVectorContext);
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
    if (failed(lhs)) {
      return failure();
    }
    auto rhs = emitExpression(rhsValue);
    if (failed(rhs)) {
      return failure();
    }
    return (Twine("(") + *lhs + " " + operation + " " + *rhs + ")").str();
  }

  [[nodiscard]] static StringRef binaryOperator(const StringRef name) {
    return llvm::StringSwitch<StringRef>(name)
        .Cases({"arith.addi", "arith.addf"}, "+")
        .Cases({"arith.subi", "arith.subf"}, "-")
        .Cases({"arith.muli", "arith.mulf"}, "*")
        .Cases({"arith.divsi", "arith.divui", "arith.divf"}, "/")
        .Cases({"arith.remsi", "arith.remui"}, "%")
        .Case("arith.andi", "&&")
        .Case("arith.ori", "||")
        .Case("arith.xori", "!=")
        .Case("arith.shli", "<<")
        .Case("arith.shrui", ">>")
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
        .Case("arith.index_cast", true)
        .Cases({"arith.sitofp", "arith.uitofp"}, true)
        .Cases({"arith.fptosi", "arith.fptoui"}, true)
        .Default(false);
  }

  [[nodiscard]] static std::string castTarget(const StringRef name,
                                              const Type resultType) {
    if (resultType.isF64()) {
      return "float";
    }
    if (resultType.isInteger(1)) {
      return "bool";
    }
    if (resultType.isIndex()) {
      return "int";
    }
    if (auto type = dyn_cast<IntegerType>(resultType);
        type && type.getWidth() <= 64) {
      return (Twine(name == "arith.fptoui" ? "uint[" : "int[") +
              Twine(type.getWidth()) + "]")
          .str();
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
        .Case("math.ctpop", "popcount")
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
    measurementDeclarations += "bit " + name + ";\n";
    *output << name << " = measure " << *qubit << ";\n";
    return success();
  }

  [[nodiscard]] FailureOr<std::string> localType(Value value) {
    auto kind = inferScalarKind(value);
    if (kind == "bit") {
      kind = "bool";
    }
    if (kind.empty()) {
      auto* operation = value.getDefiningOp();
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        operation = argument.getOwner()->getParentOp();
      }
      std::string type;
      llvm::raw_string_ostream stream(type);
      stream << value.getType();
      static_cast<void>(
          fail(operation, "unsupported carried scalar type " + type +
                              "; OpenQASM locals support bool, integers of "
                              "width 1-64, index, and f64"));
      return failure();
    }
    return kind;
  }

  [[nodiscard]] LogicalResult declareLocals(ValueRange values) {
    for (auto value : values) {
      auto type = localType(value);
      if (failed(type)) {
        return failure();
      }
      auto name = uniqueName("v", nextScalar);
      valueNames[value] = name;
      *output << *type << ' ' << name << ";\n";
    }
    return success();
  }

  [[nodiscard]] LogicalResult materialize(Value value) {
    auto type = localType(value);
    auto expression = emitExpression(value);
    if (failed(type) || failed(expression)) {
      return failure();
    }
    auto name = uniqueName("v", nextScalar);
    *output << *type << ' ' << name << " = " << *expression << ";\n";
    valueNames[value] = std::move(name);
    return success();
  }

  [[nodiscard]] LogicalResult assignEdge(ValueRange destinations,
                                         ValueRange values,
                                         Operation* terminator) {
    llvm::SaveAndRestore guard(expressionConsumer, terminator);
    SmallVector<std::string> temporaries;
    for (auto value : values) {
      auto type = localType(value);
      auto expression = emitExpression(value);
      if (failed(type) || failed(expression)) {
        return failure();
      }
      auto name = uniqueName("tmp", nextScalar);
      *output << *type << ' ' << name << " = " << *expression << ";\n";
      temporaries.push_back(std::move(name));
    }
    for (auto [destination, temporary] :
         llvm::zip_equal(destinations, temporaries)) {
      *output << valueNames.at(destination) << " = " << temporary << ";\n";
    }
    return success();
  }

  [[nodiscard]] LogicalResult emitResultBlock(Block& block,
                                              ValueRange results) {
    if (failed(emitBlock(block))) {
      return failure();
    }
    return assignEdge(results, block.getTerminator()->getOperands(),
                      block.getTerminator());
  }

  [[nodiscard]] LogicalResult emitIf(scf::IfOp ifOp) {
    if (failed(declareLocals(ifOp.getResults()))) {
      return failure();
    }
    llvm::SaveAndRestore materializeGuard(
        materializeScalars, materializeScalars || ifOp.getNumResults() != 0);
    auto condition = emitExpression(ifOp.getCondition());
    if (failed(condition)) {
      return failure();
    }
    *output << "if (" << *condition << ") {\n";
    output->indent();
    if (failed(
            emitResultBlock(ifOp.getThenRegion().front(), ifOp.getResults()))) {
      return failure();
    }
    output->unindent();
    if (!ifOp.getElseRegion().empty()) {
      *output << "} else {\n";
      output->indent();
      if (failed(emitResultBlock(ifOp.getElseRegion().front(),
                                 ifOp.getResults()))) {
        return failure();
      }
      output->unindent();
    }
    *output << "}\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitFor(scf::ForOp forOp) {
    if (failed(declareLocals(forOp.getResults())) ||
        failed(assignEdge(forOp.getResults(), forOp.getInitArgs(), forOp))) {
      return failure();
    }
    for (auto [argument, result] :
         llvm::zip_equal(forOp.getRegionIterArgs(), forOp.getResults())) {
      valueNames[argument] = valueNames.at(result);
    }
    llvm::SaveAndRestore materializeGuard(
        materializeScalars, materializeScalars || forOp.getNumResults() != 0);
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
    if (failed(emitResultBlock(*forOp.getBody(), forOp.getResults()))) {
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
    const bool ordinary =
        whileOp.getInits().empty() && whileOp.getNumResults() == 0 &&
        llvm::all_of(before.without_terminator(), [](Operation& operation) {
          return isInlineExpressionOperation(operation) &&
                 (isa<cbit::LoadOp, cbit::ReadOp>(&operation) ||
                  isMemoryEffectFree(&operation));
        });
    if (!ordinary) {
      if (failed(declareLocals(before.getArguments())) ||
          failed(declareLocals(whileOp.getResults())) ||
          failed(
              assignEdge(before.getArguments(), whileOp.getInits(), whileOp))) {
        return failure();
      }
      for (auto [argument, result] :
           llvm::zip_equal(after.getArguments(), whileOp.getResults())) {
        valueNames[argument] = valueNames.at(result);
      }
      llvm::SaveAndRestore materializeGuard(materializeScalars, true);
      *output << "while (true) {\n";
      output->indent();
      if (failed(emitBlock(before))) {
        return failure();
      }
      llvm::SaveAndRestore consumerGuard(expressionConsumer,
                                         conditionOp.getOperation());
      auto condition = emitExpression(conditionOp.getCondition());
      if (failed(condition)) {
        return failure();
      }
      const auto conditionName = uniqueName("cond", nextScalar);
      *output << "bool " << conditionName << " = " << *condition << ";\n";
      if (failed(assignEdge(whileOp.getResults(), conditionOp.getArgs(),
                            conditionOp))) {
        return failure();
      }
      *output << "if (!" << conditionName << ") {\n";
      output->indent();
      *output << "break;\n";
      output->unindent();
      *output << "}\n";
      if (failed(emitBlock(after)) ||
          failed(assignEdge(before.getArguments(), yieldOp.getResults(),
                            yieldOp))) {
        return failure();
      }
      output->unindent();
      *output << "}\n";
      return success();
    }
    llvm::SaveAndRestore consumerGuard(expressionConsumer,
                                       conditionOp.getOperation());
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
    if (failed(declareLocals(switchOp.getResults()))) {
      return failure();
    }
    llvm::SaveAndRestore materializeGuard(materializeScalars,
                                          materializeScalars ||
                                              switchOp.getNumResults() != 0);
    auto argument = emitExpression(switchOp.getArg());
    if (failed(argument)) {
      return failure();
    }

    const auto cases = switchOp.getCases();
    if (cases.empty()) {
      return emitResultBlock(switchOp.getDefaultBlock(), switchOp.getResults());
    }
    *output << "switch (" << *argument << ") {\n";
    output->indent();
    for (const auto [index, caseValue] : llvm::enumerate(cases)) {
      *output << "case " << caseValue << " {\n";
      output->indent();
      if (failed(emitResultBlock(
              switchOp.getCaseBlock(static_cast<unsigned>(index)),
              switchOp.getResults()))) {
        return failure();
      }
      output->unindent();
      *output << "}\n";
    }
    *output << "default {\n";
    output->indent();
    if (failed(emitResultBlock(switchOp.getDefaultBlock(),
                               switchOp.getResults()))) {
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
      if (auto integer = dyn_cast<IntegerType>(value.getType());
          integer && integer.getWidth() > 1) {
        *expression = scalarOutputs[scalarIndex].kind + "(" + *expression + ")";
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
