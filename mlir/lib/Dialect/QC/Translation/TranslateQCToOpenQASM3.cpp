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
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/OpenQASMAttributes.h"
#include "mlir/Dialect/Utils/AngleConversion.h"
#include "mlir/Dialect/Utils/Utils.h"
#include "mlir/Target/OpenQASM/GateCatalog.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
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
#include <iterator>
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
};

struct ScalarOutput {
  Value value;
  std::string name;
  std::string kind;
  std::optional<uint32_t> angleWidth;
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

[[nodiscard]] static std::optional<int64_t>
getConstantInteger(const Value value) {
  return getConstantIntValue(value);
}

namespace {

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
  DenseSet<Value> returnedMemrefs;
  SmallVector<ScalarOutput> scalarOutputs;
  SmallVector<ScalarOutput> scalarInputs;
  DenseMap<Value, uint32_t> angleValues;
  DenseSet<Value> uintValues;
  DenseSet<Value> bitValues;
  DenseSet<Operation*> canonicalAngleOperations;
  std::optional<uint32_t> finalGatePrecision;
  llvm::StringSet<> usedNames;
  llvm::StringSet<> fixedHelpers;
  SmallVector<std::string> compositeHelpers;
  size_t nextQubit = 0;
  size_t nextBit = 0;
  size_t nextScalar = 0;
  size_t nextLoop = 0;
  size_t nextHelper = 0;

  struct SafeShift {
    Value lhs;
    Value rhs;
    StringRef operation;
    arith::SelectOp distanceSelect;
    arith::SelectOp resultSelect;
    Operation* shift = nullptr;
    Operation* result = nullptr;
  };

  struct CarriedVariable {
    std::string name;
    std::string kind;
    std::optional<uint32_t> angleWidth;
    bool unsignedInteger = false;
    bool bit = false;
  };

  [[nodiscard]] static LogicalResult fail(Operation* operation,
                                          const Twine& message) {
    operation->emitError() << "OpenQASM emission error: " << message;
    return failure();
  }

  [[nodiscard]] static FailureOr<std::string>
  failExpression(const Value value, const Twine& message) {
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

  [[nodiscard]] FailureOr<ScalarOutput>
  scalarInterface(const DictionaryAttr metadata, const Value value,
                  const StringRef expectedDirection) {
    const auto failInterface =
        [&](const Twine& message) -> FailureOr<ScalarOutput> {
      emitError(value.getLoc()) << "OpenQASM emission error: " << message;
      return failure();
    };
    if (!metadata) {
      return failInterface("missing OpenQASM scalar interface metadata");
    }
    const auto kind = metadata.getAs<StringAttr>("kind");
    const auto name = metadata.getAs<StringAttr>("name");
    if (!kind || !name) {
      return failInterface("malformed OpenQASM scalar interface metadata");
    }

    std::string declarationKind;
    std::optional<uint32_t> angleWidth;
    const auto type = value.getType();
    if (kind.getValue() == "angle") {
      const auto integerType = dyn_cast<IntegerType>(type);
      if (!integerType ||
          !mqt::angle::isSupportedWidth(integerType.getWidth())) {
        return failInterface(
            "angle interface metadata does not match its type");
      }
      const auto bitWidth = integerType.getWidth();
      declarationKind = (Twine("angle[") + Twine(bitWidth) + "]").str();
      angleWidth = bitWidth;
      if (expectedDirection == "input") {
        angleValues[value] = bitWidth;
      }
    } else if (kind.getValue() == "uint") {
      const auto integerType = dyn_cast<IntegerType>(type);
      if (!integerType ||
          !mqt::angle::isSupportedWidth(integerType.getWidth())) {
        return failInterface("uint interface metadata does not match its type");
      }
      const auto bitWidth = integerType.getWidth();
      declarationKind = (Twine("uint[") + Twine(bitWidth) + "]").str();
      if (expectedDirection == "input") {
        uintValues.insert(value);
      }
    } else if (kind.getValue() == "int") {
      if (!type.isInteger(64)) {
        return failInterface("int interface metadata does not match its type");
      }
      declarationKind = "int";
    } else if (kind.getValue() == "float") {
      if (!type.isF64()) {
        return failInterface(
            "float interface metadata does not match its type");
      }
      declarationKind = "float";
    } else if (kind.getValue() == "bool") {
      if (!type.isInteger(1)) {
        return failInterface("bool interface metadata does not match its type");
      }
      declarationKind = "bool";
    } else {
      return failInterface("unknown OpenQASM scalar interface kind");
    }

    std::string interfaceName;
    if (expectedDirection == "output") {
      interfaceName = outputName(name.getValue());
    } else if (isValidOutputName(name.getValue()) &&
               usedNames.insert(name.getValue()).second) {
      interfaceName = name.getValue().str();
    } else {
      interfaceName = uniqueName("in", nextScalar);
    }
    return ScalarOutput{.value = value,
                        .name = std::move(interfaceName),
                        .kind = std::move(declarationKind),
                        .angleWidth = angleWidth};
  }

  [[nodiscard]] std::string qubitRegisterName(const StringRef requested) {
    if (isValidOutputName(requested) && usedNames.insert(requested).second) {
      return requested.str();
    }
    return uniqueName("q", nextQubit);
  }

  [[nodiscard]] LogicalResult preflight() {
    if (const auto rawPrecision =
            moduleOp->getAttr(mqt::angle::FINAL_QUANTIZATION_ATTR)) {
      const auto precision = dyn_cast<IntegerAttr>(rawPrecision);
      if (!precision || precision.getValue().isZero() ||
          precision.getValue().ugt(mqt::angle::MACHINE_WIDTH)) {
        return fail(moduleOp, "invalid final gate-angle precision metadata");
      }
      finalGatePrecision =
          static_cast<uint32_t>(precision.getValue().getZExtValue());
    }
    SmallVector<func::FuncOp> functions(moduleOp.getOps<func::FuncOp>());
    if (functions.size() != 1) {
      return fail(moduleOp, "expected exactly one function");
    }
    function = functions.front();
    if (function.isExternal() || function.getBody().getBlocks().size() != 1) {
      return fail(function,
                  "expected one defined function with one entry block");
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
    collectCanonicalAngleOperations();
    return success();
  }

  [[nodiscard]] LogicalResult collectProgramShape() {
    for (const auto [index, argument] :
         llvm::enumerate(function.getArguments())) {
      auto metadata = dyn_cast_or_null<DictionaryAttr>(
          function.getArgAttr(index, openqasm::SCALAR_ATTR));
      auto input = scalarInterface(metadata, argument, "input");
      if (failed(input)) {
        return failure();
      }
      valueNames[argument] = input->name;
      scalarInputs.push_back(std::move(*input));
    }

    auto returnOp =
        dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
    if (!returnOp) {
      return fail(function, "entry block must end in func.return");
    }
    for (const auto [index, value] : llvm::enumerate(returnOp.getOperands())) {
      if (returnOp.getNumOperands() == 1 && isCanonicalStatus(value, index)) {
        continue;
      }
      if (isa<MemRefType>(value.getType())) {
        returnedMemrefs.insert(value);
        continue;
      }
      auto metadata = dyn_cast_or_null<DictionaryAttr>(
          function.getResultAttr(index, openqasm::SCALAR_ATTR));
      if (metadata) {
        auto interface = scalarInterface(metadata, value, "output");
        if (failed(interface)) {
          return failure();
        }
        scalarOutputs.push_back(std::move(*interface));
      } else {
        auto kind = inferScalarKind(value);
        if (kind.empty()) {
          return fail(returnOp, "unsupported scalar output type for function "
                                "result " +
                                    Twine(index));
        }
        scalarOutputs.push_back(
            {.value = value, .name = outputName({}), .kind = std::move(kind)});
      }
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
        StringRef requested;
        if (const auto attr = alloc->getAttrOfType<StringAttr>(
                utils::QUBIT_REGISTER_NAME_ATTR)) {
          requested = attr.getValue();
        }
        resource.name = qubitRegisterName(requested);
      } else {
        resource.output = returnedMemrefs.contains(alloc.getResult());
        StringRef requested;
        if (const auto attr = alloc->getAttrOfType<StringAttr>(
                utils::CLASSICAL_REGISTER_NAME_ATTR)) {
          requested = attr.getValue();
        }
        resource.name =
            resource.output ? outputName(requested) : uniqueName("c", nextBit);
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
    }
    return success();
  }

  [[nodiscard]] static bool isCanonicalStatus(const Value value,
                                              const size_t resultIndex) {
    if (resultIndex != 0 || !value.getType().isInteger(64)) {
      return false;
    }
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    auto integer =
        constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
    return integer && integer.getValue().isZero();
  }

  [[nodiscard]] bool isBitValue(const Value value) const {
    if (!value.getType().isInteger(1)) {
      return false;
    }
    if (bitValues.contains(value) || value.getDefiningOp<qc::MeasureOp>()) {
      return true;
    }
    auto load = value.getDefiningOp<memref::LoadOp>();
    if (!load) {
      return false;
    }
    const auto resource = resources.find(load.getMemRef());
    return resource != resources.end() &&
           resource->second.kind == ResourceKind::Bit;
  }

  [[nodiscard]] std::string inferScalarKind(const Value value) const {
    const auto type = value.getType();
    if (type.isInteger(1)) {
      return isBitValue(value) ? "bit" : "bool";
    }
    if (type.isInteger(64) || type.isIndex()) {
      return "int";
    }
    if (const auto integer = dyn_cast<IntegerType>(type);
        integer && integer.getWidth() <= mqt::angle::MACHINE_WIDTH) {
      return (Twine("uint[") + Twine(integer.getWidth()) + "]").str();
    }
    if (type.isF64()) {
      return "float";
    }
    return {};
  }

  void mapCarriedValue(const Value value, const CarriedVariable& variable) {
    valueNames[value] = variable.name;
    if (variable.angleWidth) {
      angleValues[value] = *variable.angleWidth;
    } else if (variable.unsignedInteger) {
      uintValues.insert(value);
    } else if (variable.bit) {
      bitValues.insert(value);
    }
  }

  [[nodiscard]] FailureOr<std::string>
  emitCarriedValue(const Value value, const CarriedVariable& variable) {
    if (variable.angleWidth) {
      return emitAngleUse(value, *variable.angleWidth);
    }
    if (variable.unsignedInteger) {
      return emitUnsignedOperand(value);
    }
    return emitExpression(value);
  }

  [[nodiscard]] FailureOr<SmallVector<CarriedVariable>>
  declareCarriedVariables(const ValueRange results,
                          const ArrayRef<SmallVector<Value>> sources,
                          const ValueRange initialValues = {}) {
    if (results.size() != sources.size() ||
        (!initialValues.empty() && initialValues.size() != results.size())) {
      return failure();
    }
    SmallVector<CarriedVariable> variables;
    variables.reserve(results.size());
    for (const auto [index, result] : llvm::enumerate(results)) {
      std::optional<uint32_t> angle;
      bool isUnsigned = false;
      bool isBit = false;
      for (const auto source : sources[index]) {
        if (!angle) {
          angle = angleWidth(source);
        }
        isUnsigned |= isUnsignedValue(source);
        isBit |= isBitValue(source);
      }
      auto kind = inferScalarKind(result);
      if (angle) {
        const auto integer = dyn_cast<IntegerType>(result.getType());
        if (!integer || integer.getWidth() != *angle) {
          return failure();
        }
        kind = (Twine("angle[") + Twine(*angle) + "]").str();
        isUnsigned = false;
      } else if (isUnsigned) {
        const auto integer = dyn_cast<IntegerType>(result.getType());
        if (!integer || !mqt::angle::isSupportedWidth(integer.getWidth())) {
          return failure();
        }
        kind = (Twine("uint[") + Twine(integer.getWidth()) + "]").str();
      } else if (isBit) {
        if (!result.getType().isInteger(1)) {
          return failure();
        }
        kind = "bit";
      }
      if (kind.empty()) {
        return failure();
      }
      CarriedVariable variable{.name = uniqueName("s", nextScalar),
                               .kind = std::move(kind),
                               .angleWidth = angle,
                               .unsignedInteger = isUnsigned,
                               .bit = isBit};
      *output << variable.kind << ' ' << variable.name;
      if (!initialValues.empty()) {
        auto initializer = emitCarriedValue(initialValues[index], variable);
        if (failed(initializer)) {
          return failure();
        }
        *output << " = " << *initializer;
      }
      *output << ";\n";
      mapCarriedValue(result, variable);
      variables.push_back(std::move(variable));
    }
    return variables;
  }

  [[nodiscard]] LogicalResult
  emitCarriedAssignments(const ValueRange values,
                         const ArrayRef<CarriedVariable> variables) {
    if (values.size() != variables.size()) {
      return failure();
    }
    SmallVector<std::string> stagedNames;
    stagedNames.reserve(values.size());
    for (const auto [value, variable] : llvm::zip_equal(values, variables)) {
      auto expression = emitCarriedValue(value, variable);
      if (failed(expression)) {
        return failure();
      }
      auto staged = uniqueName("next", nextScalar);
      *output << variable.kind << ' ' << staged << " = " << *expression
              << ";\n";
      stagedNames.push_back(std::move(staged));
    }
    for (const auto [variable, staged] :
         llvm::zip_equal(variables, stagedNames)) {
      *output << variable.name << " = " << staged << ";\n";
    }
    return success();
  }

  [[nodiscard]] LogicalResult emitDeclarations() {
    for (const auto& scalar : scalarInputs) {
      *output << "input " << scalar.kind << ' ' << scalar.name << ";\n";
    }
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
    if (!scalarInputs.empty() || !resourceOrder.empty() ||
        !scalarOutputs.empty()) {
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
      return (canonicalAngleOperations.contains(&operation) ||
              isSafeShiftMember(operation))
                 ? success()
                 : fail(&operation, "arith.select is not supported");
    }
    if (isa<arith::ConstantOp, memref::LoadOp, memref::AllocOp,
            memref::DeallocOp, qc::AllocOp, qc::DeallocOp, qc::StaticOp>(
            &operation)) {
      return success();
    }
    if (isInlineExpressionOperation(operation)) {
      if (llvm::all_of(operation.getResults(),
                       [](const Value result) { return result.use_empty(); })) {
        return success();
      }
      return validateInlineExpressionOperation(operation);
    }
    if (auto assertion = dyn_cast<cf::AssertOp>(&operation);
        assertion && isUnsignedDivisionSafetyAssert(assertion)) {
      return success();
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
    return isa<arith::ConstantOp, arith::CmpIOp, arith::CmpFOp, arith::SelectOp,
               arith::BitcastOp, math::CtPopOp, LLVM::FshlOp, LLVM::FshrOp>(
               &operation) ||
           !binaryOperator(name).empty() || name == "arith.negf" ||
           name == "arith.remf" || isScalarCast(name) ||
           !mathFunction(name).empty();
  }

  [[nodiscard]] static bool
  isUnsignedDivisionSafetyAssert(cf::AssertOp assertion) {
    const auto message = assertion.getMsg();
    if (message != "division by zero" && message != "modulo by zero") {
      return false;
    }
    Value divisor;
    if (auto comparison = assertion.getArg().getDefiningOp<arith::CmpIOp>()) {
      if (comparison.getPredicate() != arith::CmpIPredicate::ne) {
        return false;
      }
      if (getConstantInteger(comparison.getRhs()) == 0) {
        divisor = comparison.getLhs();
      } else if (getConstantInteger(comparison.getLhs()) == 0) {
        divisor = comparison.getRhs();
      } else {
        return false;
      }
    } else if (assertion.getArg().getType().isInteger(1)) {
      divisor = assertion.getArg();
    } else {
      return false;
    }
    return llvm::any_of(divisor.getUsers(), [&](Operation* user) {
      if (message == "division by zero") {
        auto division = dyn_cast<arith::DivUIOp>(user);
        return division && division.getRhs() == divisor;
      }
      auto remainder = dyn_cast<arith::RemUIOp>(user);
      return remainder && remainder.getRhs() == divisor;
    });
  }

  [[nodiscard]] LogicalResult
  validateInlineExpressionOperation(Operation& operation) {
    if (canonicalAngleOperations.contains(&operation) ||
        isCanonicalAngleBridgeMember(operation) ||
        isSafeShiftMember(operation)) {
      return success();
    }
    for (const auto result : operation.getResults()) {
      const auto type = result.getType();
      const auto integer = dyn_cast<IntegerType>(type);
      if ((!integer || integer.getWidth() > mqt::angle::MACHINE_WIDTH) &&
          !type.isIndex() && !type.isF64()) {
        return fail(&operation,
                    "unsupported scalar expression result type on '" +
                        operation.getName().getStringRef() + "'");
      }
      if (failed(emitExpression(result))) {
        return failure();
      }
    }
    return success();
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
    return (Twine(resource->second.name) + "[" + Twine(*index) + "]").str();
  }

  void collectCanonicalAngleOperations() {
    function.walk([&](Operation* operation) {
      if (!operation->hasAttrOfType<UnitAttr>(openqasm::ANGLE_VALUE_ATTR)) {
        return;
      }
      for (const auto result : operation->getResults()) {
        const auto integerType = dyn_cast<IntegerType>(result.getType());
        if (!integerType) {
          continue;
        }
        const auto resize = mqt::angle::matchResize(result);
        if (!resize || resize->targetWidth != integerType.getWidth()) {
          continue;
        }
        canonicalAngleOperations.insert(resize->operations.begin(),
                                        resize->operations.end());
      }
    });

    function.walk([&](Operation* operation) {
      for (const auto result : operation->getResults()) {
        const auto source = mqt::angle::matchFloatToBits(result);
        if (!source) {
          continue;
        }
        SmallVector<Value> worklist{result};
        DenseSet<Value> visited;
        while (!worklist.empty()) {
          const auto value = worklist.pop_back_val();
          if (value == *source || !visited.insert(value).second) {
            continue;
          }
          auto* definingOp = value.getDefiningOp();
          if (definingOp == nullptr) {
            continue;
          }
          canonicalAngleOperations.insert(definingOp);
          llvm::append_range(worklist, definingOp->getOperands());
        }
      }
    });
  }

  [[nodiscard]] static std::optional<SafeShift>
  matchSafeShift(const Value value) {
    Value selected = value;
    Operation* result = value.getDefiningOp();
    if (auto truncation = value.getDefiningOp<arith::TruncIOp>()) {
      selected = truncation.getIn();
    }
    auto resultSelect = selected.getDefiningOp<arith::SelectOp>();
    if (!resultSelect || !getConstantInteger(resultSelect.getFalseValue()) ||
        *getConstantInteger(resultSelect.getFalseValue()) != 0) {
      return std::nullopt;
    }

    auto* shift = resultSelect.getTrueValue().getDefiningOp();
    StringRef operation;
    if (isa_and_nonnull<arith::ShLIOp>(shift)) {
      operation = "<<";
    } else if (isa_and_nonnull<arith::ShRUIOp>(shift)) {
      operation = ">>";
    } else {
      return std::nullopt;
    }
    auto lhs = shift->getOperand(0);
    auto safeDistance = shift->getOperand(1);
    auto distanceSelect = safeDistance.getDefiningOp<arith::SelectOp>();
    if (!distanceSelect ||
        distanceSelect.getCondition() != resultSelect.getCondition() ||
        !getConstantInteger(distanceSelect.getFalseValue()) ||
        *getConstantInteger(distanceSelect.getFalseValue()) != 0) {
      return std::nullopt;
    }

    auto comparison =
        resultSelect.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!comparison || comparison.getPredicate() != arith::CmpIPredicate::ult ||
        comparison.getLhs() != distanceSelect.getTrueValue()) {
      return std::nullopt;
    }
    const auto limit = getConstantInteger(comparison.getRhs());
    if (!limit || *limit <= 0) {
      return std::nullopt;
    }

    if (auto extension = lhs.getDefiningOp<arith::ExtUIOp>()) {
      lhs = extension.getIn();
    }
    auto rhs = distanceSelect.getTrueValue();
    if (auto extension = rhs.getDefiningOp<arith::ExtUIOp>()) {
      rhs = extension.getIn();
    }
    const auto lhsType = dyn_cast<IntegerType>(lhs.getType());
    if (!lhsType || std::cmp_not_equal(lhsType.getWidth(), *limit)) {
      return std::nullopt;
    }
    return SafeShift{.lhs = lhs,
                     .rhs = rhs,
                     .operation = operation,
                     .distanceSelect = distanceSelect,
                     .resultSelect = resultSelect,
                     .shift = shift,
                     .result = result};
  }

  [[nodiscard]] static bool isSafeShiftMember(Operation& operation) {
    const auto matchesOperation = [&](SafeShift shift) {
      return shift.distanceSelect.getOperation() == &operation ||
             shift.resultSelect.getOperation() == &operation ||
             shift.shift == &operation || shift.result == &operation;
    };
    for (const auto result : operation.getResults()) {
      if (const auto shift = matchSafeShift(result);
          shift && matchesOperation(*shift)) {
        return true;
      }
      for (Operation* user : result.getUsers()) {
        for (const auto userResult : user->getResults()) {
          if (const auto shift = matchSafeShift(userResult);
              shift && matchesOperation(*shift)) {
            return true;
          }
          for (Operation* finalUser : userResult.getUsers()) {
            for (const auto finalResult : finalUser->getResults()) {
              if (const auto shift = matchSafeShift(finalResult);
                  shift && matchesOperation(*shift)) {
                return true;
              }
            }
          }
        }
      }
    }
    return false;
  }

  [[nodiscard]] static bool isCanonicalAngleBridgeMember(Operation& operation) {
    const auto isCanonicalResult = [](const Value value) {
      return mqt::angle::matchQuantizedRadians(value).has_value();
    };
    if (llvm::any_of(operation.getResults(), isCanonicalResult)) {
      return true;
    }
    if (!isa<arith::UIToFPOp>(operation)) {
      return false;
    }
    for (const auto result : operation.getResults()) {
      for (Operation* user : result.getUsers()) {
        if (llvm::any_of(user->getResults(), isCanonicalResult)) {
          return true;
        }
      }
    }
    return false;
  }

  [[nodiscard]] std::optional<uint32_t> angleWidth(const Value value) const {
    if (const auto known = angleValues.find(value);
        known != angleValues.end()) {
      return known->second;
    }
    const auto integerType = dyn_cast<IntegerType>(value.getType());
    auto* definingOp = value.getDefiningOp();
    if (!integerType || definingOp == nullptr) {
      return std::nullopt;
    }
    if (!definingOp->hasAttrOfType<UnitAttr>(openqasm::ANGLE_VALUE_ATTR)) {
      return std::nullopt;
    }
    return integerType.getWidth();
  }

  [[nodiscard]] static bool isAngleOperand(Operation* operation,
                                           const int32_t position) {
    const auto operands = operation->getAttrOfType<DenseI32ArrayAttr>(
        openqasm::ANGLE_OPERANDS_ATTR);
    return operands && llvm::is_contained(operands.asArrayRef(), position);
  }

  [[nodiscard]] bool isUnsignedValue(const Value value) const {
    if (uintValues.contains(value)) {
      return true;
    }
    auto* definingOp = value.getDefiningOp();
    return definingOp != nullptr &&
           definingOp->hasAttrOfType<UnitAttr>(openqasm::UINT_VALUE_ATTR);
  }

  [[nodiscard]] FailureOr<std::string>
  emitSourceOperand(const Value value, const bool unsignedIntegers = false) {
    if (const auto width = angleWidth(value)) {
      return emitAngleUse(value, *width);
    }
    if (isUnsignedValue(value)) {
      return emitExpression(value, /*unsignedIntegers=*/true);
    }
    return emitExpression(value, unsignedIntegers);
  }

  [[nodiscard]] FailureOr<std::string> emitUnsignedOperand(const Value value) {
    const auto width = angleWidth(value);
    if (!width) {
      return emitExpression(value, /*unsignedIntegers=*/true);
    }
    auto expression = emitAngleUse(value, *width);
    if (failed(expression)) {
      return failure();
    }
    return (Twine("uint[") + Twine(*width) + "](bit[" + Twine(*width) + "](" +
            *expression + "))")
        .str();
  }

  [[nodiscard]] FailureOr<std::string> emitRotation(const Value value) {
    auto* operation = value.getDefiningOp();
    if (operation == nullptr || operation->getNumOperands() != 3 ||
        operation->getOperand(0) != operation->getOperand(1)) {
      return failExpression(value,
                            "only funnel shifts representing rotations are "
                            "supported");
    }
    auto operand =
        angleWidth(value)
            ? emitAngleUse(operation->getOperand(0),
                           cast<IntegerType>(value.getType()).getWidth())
            : emitUnsignedOperand(operation->getOperand(0));
    auto distance = emitExpression(operation->getOperand(2),
                                   /*unsignedIntegers=*/true);
    if (failed(operand) || failed(distance)) {
      return failure();
    }
    return (Twine(isa<LLVM::FshlOp>(operation) ? "rotl(" : "rotr(") + *operand +
            ", int(" + *distance + "))")
        .str();
  }

  [[nodiscard]] FailureOr<std::string> emitAngleUse(const Value bits,
                                                    const uint32_t bitWidth) {
    if (const auto known = angleValues.find(bits);
        known != angleValues.end() && known->second == bitWidth) {
      return emitExpression(bits);
    }
    if (const auto resize = mqt::angle::matchResize(bits);
        resize && resize->targetWidth == bitWidth) {
      auto expression = emitAngleUse(resize->source, resize->sourceWidth);
      if (failed(expression)) {
        return failure();
      }
      return (Twine("angle[") + Twine(bitWidth) + "](" + *expression + ")")
          .str();
    }
    if (const auto shift = matchSafeShift(bits)) {
      return emitBinary(shift->lhs, shift->operation, shift->rhs,
                        /*unsignedIntegers=*/true,
                        /*lhsAngle=*/true, /*rhsAngle=*/false);
    }
    if (const auto radians = mqt::angle::matchFloatToBits(bits)) {
      auto expression = emitExpression(*radians);
      if (failed(expression)) {
        return failure();
      }
      return (Twine("angle[") + Twine(bitWidth) + "](" + *expression + ")")
          .str();
    }
    if (auto* operation = bits.getDefiningOp();
        operation != nullptr && isa<LLVM::FshlOp, LLVM::FshrOp>(operation)) {
      return emitRotation(bits);
    }
    if (auto* operation = bits.getDefiningOp()) {
      if (operation->hasAttrOfType<UnitAttr>(openqasm::ANGLE_VALUE_ATTR) &&
          operation->getNumOperands() == 2 &&
          !binaryOperator(operation->getName().getStringRef()).empty()) {
        auto emittedOperator =
            binaryOperator(operation->getName().getStringRef());
        if (operation->getName().getStringRef() == "arith.andi") {
          emittedOperator = "&";
        } else if (operation->getName().getStringRef() == "arith.ori") {
          emittedOperator = "|";
        } else if (operation->getName().getStringRef() == "arith.xori") {
          emittedOperator = "^";
        }
        if (operation->getName().getStringRef() == "arith.xori") {
          llvm::APInt constant;
          Value operand;
          if (matchPattern(operation->getOperand(0),
                           m_ConstantInt(&constant)) &&
              constant.isAllOnes()) {
            operand = operation->getOperand(1);
          } else if (matchPattern(operation->getOperand(1),
                                  m_ConstantInt(&constant)) &&
                     constant.isAllOnes()) {
            operand = operation->getOperand(0);
          }
          if (operand) {
            auto expression = emitAngleUse(operand, bitWidth);
            if (failed(expression)) {
              return failure();
            }
            return (Twine("(~") + *expression + ")").str();
          }
        }
        if (operation->getName().getStringRef() == "arith.subi" &&
            getConstantInteger(operation->getOperand(0)) == 0) {
          auto operand = emitAngleUse(operation->getOperand(1), bitWidth);
          if (failed(operand)) {
            return failure();
          }
          return (Twine("(-") + *operand + ")").str();
        }
        return emitBinary(
            operation->getOperand(0), emittedOperator, operation->getOperand(1),
            /*unsignedIntegers=*/true, isAngleOperand(operation, 0),
            isAngleOperand(operation, 1));
      }
    }
    auto expression = emitExpression(bits, /*unsignedIntegers=*/true);
    if (failed(expression)) {
      return failure();
    }
    if (isUnsignedValue(bits)) {
      return (Twine("angle[") + Twine(bitWidth) + "](bit[" + Twine(bitWidth) +
              "](" + *expression + "))")
          .str();
    }
    return (Twine("angle[") + Twine(bitWidth) + "](bit[" + Twine(bitWidth) +
            "](uint[" + Twine(bitWidth) + "](" + *expression + ")))")
        .str();
  }

  [[nodiscard]] FailureOr<std::string>
  emitExpression(const Value value, const bool unsignedIntegers = false) {
    if (const auto found = valueNames.find(value); found != valueNames.end()) {
      return found->second;
    }
    if (const auto quantized = mqt::angle::matchQuantizedRadians(value)) {
      return emitAngleUse(quantized->bits, quantized->bitWidth);
    }
    if (const auto shift = matchSafeShift(value)) {
      return emitBinary(shift->lhs, shift->operation, shift->rhs,
                        /*unsignedIntegers=*/true,
                        /*lhsAngle=*/angleWidth(value).has_value(),
                        /*rhsAngle=*/false);
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
      const auto integerType = dyn_cast<IntegerType>(value.getType());
      const bool printUnsigned =
          unsignedIntegers ||
          (integerType && integerType.getWidth() != 1 &&
           integerType.getWidth() != mqt::angle::MACHINE_WIDTH);
      return emitConstant(constant, printUnsigned);
    }
    if (isa<ub::PoisonOp>(operation)) {
      return failExpression(value, "poison values are not supported");
    }
    if (value.getType().isInteger(1) &&
        operation->hasAttrOfType<UnitAttr>(openqasm::BIT_EXTRACT_ATTR)) {
      Value source;
      uint64_t bit = 0;
      if (auto truncation = dyn_cast<arith::TruncIOp>(operation)) {
        source = truncation.getIn();
      } else if (auto mask = dyn_cast<arith::AndIOp>(operation)) {
        if (getConstantInteger(mask.getLhs()) == 1) {
          source = mask.getRhs();
        } else if (getConstantInteger(mask.getRhs()) == 1) {
          source = mask.getLhs();
        }
      }
      if (!source) {
        return failExpression(value, "malformed scalar bit extraction");
      }
      if (auto shift = source.getDefiningOp<arith::ShRUIOp>()) {
        const auto distance = getConstantInteger(shift.getRhs());
        if (!distance || *distance < 0) {
          return failExpression(value, "dynamic scalar bit extraction is not "
                                       "supported");
        }
        source = shift.getLhs();
        bit = static_cast<uint64_t>(*distance);
      }
      const auto sourceType = dyn_cast<IntegerType>(source.getType());
      if (sourceType && bit < sourceType.getWidth()) {
        auto expression = emitSourceOperand(source,
                                            /*unsignedIntegers=*/true);
        if (failed(expression)) {
          return failure();
        }
        return (Twine(*expression) + "[" + Twine(bit) + "]").str();
      }
    }
    if (auto cmp = dyn_cast<arith::CmpIOp>(operation)) {
      if (cmp.getLhs().getType().isInteger(1) &&
          (cmp.getPredicate() == arith::CmpIPredicate::eq ||
           cmp.getPredicate() == arith::CmpIPredicate::ne)) {
        const auto booleanConstant =
            [](const Value candidate) -> std::optional<bool> {
          auto constant = candidate.getDefiningOp<arith::ConstantOp>();
          const auto integer = constant
                                   ? dyn_cast<IntegerAttr>(constant.getValue())
                                   : IntegerAttr{};
          if (!integer || !integer.getType().isInteger(1)) {
            return std::nullopt;
          }
          return !integer.getValue().isZero();
        };
        Value operand;
        std::optional<bool> constant;
        if (const auto rhs = booleanConstant(cmp.getRhs())) {
          operand = cmp.getLhs();
          constant = rhs;
        } else if (const auto lhs = booleanConstant(cmp.getLhs())) {
          operand = cmp.getRhs();
          constant = lhs;
        }
        if (operand && constant) {
          auto expression = emitExpression(operand);
          if (failed(expression)) {
            return failure();
          }
          const bool preserve =
              (cmp.getPredicate() == arith::CmpIPredicate::eq) == *constant;
          return preserve ? *expression
                          : (Twine("(!") + *expression + ")").str();
        }
      }
      auto predicate = integerPredicate(cmp.getPredicate());
      if (predicate.empty()) {
        return failExpression(value, "unsupported integer comparison");
      }
      const auto unsignedPredicate =
          cmp.getPredicate() == arith::CmpIPredicate::ult ||
          cmp.getPredicate() == arith::CmpIPredicate::ule ||
          cmp.getPredicate() == arith::CmpIPredicate::ugt ||
          cmp.getPredicate() == arith::CmpIPredicate::uge;
      return emitBinary(cmp.getLhs(), predicate, cmp.getRhs(),
                        unsignedPredicate, isAngleOperand(operation, 0),
                        isAngleOperand(operation, 1));
    }
    if (auto cmp = dyn_cast<arith::CmpFOp>(operation)) {
      auto predicate = floatPredicate(cmp.getPredicate());
      if (predicate.empty()) {
        return failExpression(value, "unsupported floating-point comparison");
      }
      return emitBinary(cmp.getLhs(), predicate, cmp.getRhs());
    }
    if (isa<math::CtPopOp>(operation)) {
      auto operand = emitSourceOperand(operation->getOperand(0),
                                       /*unsignedIntegers=*/true);
      if (failed(operand)) {
        return failure();
      }
      return (Twine("popcount(") + *operand + ")").str();
    }
    if (isa<LLVM::FshlOp, LLVM::FshrOp>(operation)) {
      return emitRotation(value);
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
      auto emittedOperator = binary;
      if (!value.getType().isInteger(1) || unsignedIntegers ||
          isUnsignedValue(value)) {
        if (name == "arith.andi") {
          emittedOperator = "&";
        } else if (name == "arith.ori") {
          emittedOperator = "|";
        } else if (name == "arith.xori") {
          emittedOperator = "^";
        }
      }
      const bool unsignedOperands =
          unsignedIntegers || name == "arith.divui" || name == "arith.remui" ||
          name == "arith.shrui" ||
          ((name == "arith.andi" || name == "arith.ori" ||
            name == "arith.xori" || name == "arith.shli") &&
           !value.getType().isInteger(1));
      return emitBinary(operation->getOperand(0), emittedOperator,
                        operation->getOperand(1), unsignedOperands,
                        isAngleOperand(operation, 0),
                        isAngleOperand(operation, 1));
    }
    if (name == "arith.negf") {
      auto operand = emitExpression(operation->getOperand(0));
      if (failed(operand)) {
        return failure();
      }
      return (Twine("(-") + *operand + ")").str();
    }
    if (isScalarCast(name)) {
      const auto source = operation->getOperand(0);
      const bool angleResult =
          operation->hasAttrOfType<UnitAttr>(openqasm::ANGLE_VALUE_ATTR);
      auto operand = angleWidth(source) && !angleResult
                         ? emitUnsignedOperand(source)
                         : emitExpression(source, unsignedIntegers ||
                                                      name == "arith.extui" ||
                                                      name == "arith.uitofp");
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
      if (value.getType().isInteger(1) && unsignedIntegers) {
        type = "uint[1]";
      }
      if (name == "arith.extui" && bitValues.contains(source) &&
          value.getType().getIntOrFloatBitWidth() > 1) {
        return (Twine(type) + "(uint[1](" + *operand + "))").str();
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
      return (Twine(functionName) + "(" + llvm::join(arguments, ", ") + ")")
          .str();
    }
    return failExpression(value,
                          "unsupported expression operation '" + name + "'");
  }

  [[nodiscard]] static FailureOr<std::string>
  emitConstant(arith::ConstantOp constant, const bool unsignedInteger) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue())) {
      if (integer.getType().isInteger(1)) {
        return integer.getValue().isZero() ? std::string("false")
                                           : std::string("true");
      }
      llvm::SmallString<32> text;
      integer.getValue().toString(text, 10, !unsignedInteger);
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
  emitBinary(const Value lhsValue, const StringRef operation,
             const Value rhsValue, const bool unsignedIntegers = false,
             const bool lhsAngle = false, const bool rhsAngle = false) {
    auto emitOperand = [&](const Value operand, const bool forceAngle) {
      if (forceAngle) {
        const auto type = dyn_cast<IntegerType>(operand.getType());
        if (!type || !mqt::angle::isSupportedWidth(type.getWidth())) {
          return failExpression(operand, "angle operand is not a supported "
                                         "fixed-width integer");
        }
        return emitAngleUse(operand, type.getWidth());
      }
      return unsignedIntegers ? emitUnsignedOperand(operand)
                              : emitSourceOperand(operand,
                                                  /*unsignedIntegers=*/false);
    };
    auto lhs = emitOperand(lhsValue, lhsAngle);
    auto rhs = emitOperand(rhsValue, rhsAngle);
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
        .Cases("arith.remsi", "arith.remui", "%")
        .Case("arith.shli", "<<")
        .Case("arith.shrui", ">>")
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
      return "<";
    case arith::CmpIPredicate::ule:
      return "<=";
    case arith::CmpIPredicate::ugt:
      return ">";
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
    case arith::CmpFPredicate::UNE:
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
        .Case("arith.uitofp", true)
        .Case("arith.fptosi", true)
        .Case("arith.fptoui", true)
        .Case("arith.extsi", true)
        .Case("arith.extui", true)
        .Case("arith.trunci", true)
        .Default(false);
  }

  [[nodiscard]] static std::string castTarget(const StringRef name,
                                              const Type resultType) {
    if (name == "arith.sitofp" || name == "arith.uitofp" ||
        resultType.isF64()) {
      return "float";
    }
    if (resultType.isInteger(1)) {
      return "bool";
    }
    if (resultType.isInteger(64) || resultType.isIndex()) {
      return name == "arith.fptoui" || name == "arith.extui" ? "uint[64]"
                                                             : "int";
    }
    if (const auto integer = dyn_cast<IntegerType>(resultType)) {
      return (Twine("uint[") + Twine(integer.getWidth()) + "]").str();
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
    bitValues.insert(measurement.getResult());
    *output << "bit " << name << " = measure " << *qubit << ";\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitIf(scf::IfOp ifOp) {
    SmallVector<CarriedVariable> variables;
    if (ifOp.getNumResults() != 0) {
      if (ifOp.getElseRegion().empty()) {
        return fail(ifOp, "result-bearing scf.if requires an else region");
      }
      auto thenYield =
          cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
      auto elseYield =
          cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
      if (thenYield.getNumOperands() != ifOp.getNumResults() ||
          elseYield.getNumOperands() != ifOp.getNumResults()) {
        return fail(ifOp, "malformed result-bearing scf.if");
      }
      SmallVector<SmallVector<Value>> sources(ifOp.getNumResults());
      for (const auto index : llvm::seq<size_t>(0, ifOp.getNumResults())) {
        sources[index].append(
            {thenYield.getOperand(index), elseYield.getOperand(index)});
      }
      auto declared = declareCarriedVariables(ifOp.getResults(), sources);
      if (failed(declared)) {
        return fail(ifOp, "unsupported scf.if result type");
      }
      variables = std::move(*declared);
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
    if (!variables.empty() &&
        failed(emitCarriedAssignments(
            cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator())
                .getOperands(),
            variables))) {
      return failure();
    }
    output->unindent();
    if (!ifOp.getElseRegion().empty()) {
      *output << "} else {\n";
      output->indent();
      if (failed(emitBlock(ifOp.getElseRegion().front()))) {
        return failure();
      }
      if (!variables.empty() &&
          failed(emitCarriedAssignments(
              cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator())
                  .getOperands(),
              variables))) {
        return failure();
      }
      output->unindent();
    }
    *output << "}\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitFor(scf::ForOp forOp) {
    if (forOp.getInitArgs().size() != forOp.getNumResults() ||
        forOp.getRegionIterArgs().size() != forOp.getNumResults()) {
      return fail(forOp, "malformed scf.for loop-carried state");
    }
    SmallVector<CarriedVariable> variables;
    if (forOp.getNumResults() != 0) {
      auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      if (yield.getNumOperands() != forOp.getNumResults()) {
        return fail(forOp, "malformed scf.for yield");
      }
      SmallVector<SmallVector<Value>> sources(forOp.getNumResults());
      for (const auto index : llvm::seq<size_t>(0, forOp.getNumResults())) {
        sources[index].append(
            {forOp.getInitArgs()[index], yield.getOperand(index)});
      }
      auto declared = declareCarriedVariables(forOp.getResults(), sources,
                                              forOp.getInitArgs());
      if (failed(declared)) {
        return fail(forOp, "unsupported scf.for result type");
      }
      variables = std::move(*declared);
      for (const auto [argument, variable] :
           llvm::zip_equal(forOp.getRegionIterArgs(), variables)) {
        mapCarriedValue(argument, variable);
      }
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
    if (!variables.empty() &&
        failed(emitCarriedAssignments(
            cast<scf::YieldOp>(forOp.getBody()->getTerminator()).getOperands(),
            variables))) {
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
    const auto stateCount = whileOp.getInits().size();
    const auto resultCount = whileOp.getNumResults();
    if (before.getNumArguments() != stateCount ||
        yieldOp.getNumOperands() != stateCount ||
        after.getNumArguments() != resultCount ||
        conditionOp.getArgs().size() != resultCount) {
      return fail(whileOp, "malformed scf.while loop-carried state");
    }
    SmallVector<CarriedVariable> variables;
    if (stateCount != 0) {
      SmallVector<SmallVector<Value>> sources(stateCount);
      for (const auto index : llvm::seq<size_t>(0, stateCount)) {
        sources[index].append(
            {whileOp.getInits()[index], yieldOp.getOperand(index)});
      }
      auto declared = declareCarriedVariables(before.getArguments(), sources,
                                              whileOp.getInits());
      if (failed(declared)) {
        return fail(whileOp, "unsupported scf.while result type");
      }
      variables = std::move(*declared);
      for (const auto resultIndex : llvm::seq<size_t>(0, resultCount)) {
        const auto forwarded = conditionOp.getArgs()[resultIndex];
        auto* const found = llvm::find(before.getArguments(), forwarded);
        if (found == before.getArguments().end()) {
          return fail(whileOp,
                      "scf.while condition-region state updates are not "
                      "supported");
        }
        const auto stateIndex = static_cast<size_t>(
            std::distance(before.getArguments().begin(), found));
        mapCarriedValue(after.getArgument(resultIndex), variables[stateIndex]);
        mapCarriedValue(whileOp.getResult(resultIndex), variables[stateIndex]);
      }
    }
    for (Operation& operation : before.without_terminator()) {
      if (auto load = dyn_cast<memref::LoadOp>(operation)) {
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
    if (!variables.empty() &&
        failed(emitCarriedAssignments(yieldOp.getOperands(), variables))) {
      return failure();
    }
    output->unindent();
    *output << "}\n";
    return success();
  }

  [[nodiscard]] LogicalResult emitIndexSwitch(scf::IndexSwitchOp switchOp) {
    SmallVector<CarriedVariable> variables;
    if (switchOp.getNumResults() != 0) {
      SmallVector<SmallVector<Value>> sources(switchOp.getNumResults());
      const auto collectYield = [&](Block& block) {
        auto yield = cast<scf::YieldOp>(block.getTerminator());
        if (yield.getNumOperands() != switchOp.getNumResults()) {
          return failure();
        }
        for (const auto index :
             llvm::seq<size_t>(0, switchOp.getNumResults())) {
          sources[index].push_back(yield.getOperand(index));
        }
        return success();
      };
      for (Region& region : switchOp.getCaseRegions()) {
        if (failed(collectYield(region.front()))) {
          return fail(switchOp, "malformed scf.index_switch case yield");
        }
      }
      if (failed(collectYield(switchOp.getDefaultBlock()))) {
        return fail(switchOp, "malformed scf.index_switch default yield");
      }
      auto declared = declareCarriedVariables(switchOp.getResults(), sources);
      if (failed(declared)) {
        return fail(switchOp, "unsupported scf.index_switch result type");
      }
      variables = std::move(*declared);
    }
    auto argument = emitExpression(switchOp.getArg());
    if (failed(argument)) {
      return failure();
    }

    const auto cases = switchOp.getCases();
    if (cases.empty()) {
      if (failed(emitBlock(switchOp.getDefaultBlock()))) {
        return failure();
      }
      if (!variables.empty() &&
          failed(emitCarriedAssignments(
              cast<scf::YieldOp>(switchOp.getDefaultBlock().getTerminator())
                  .getOperands(),
              variables))) {
        return failure();
      }
      return success();
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
      if (!variables.empty() &&
          failed(emitCarriedAssignments(
              cast<scf::YieldOp>(
                  switchOp.getCaseBlock(static_cast<unsigned>(index))
                      .getTerminator())
                  .getOperands(),
              variables))) {
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
    if (!variables.empty() &&
        failed(emitCarriedAssignments(
            cast<scf::YieldOp>(switchOp.getDefaultBlock().getTerminator())
                .getOperands(),
            variables))) {
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
      if (isa<MemRefType>(value.getType())) {
        continue;
      }
      const auto& scalar = scalarOutputs[scalarIndex];
      FailureOr<std::string> expression;
      if (scalar.angleWidth) {
        expression = emitAngleUse(value, *scalar.angleWidth);
      } else if (scalar.kind.starts_with("uint[")) {
        expression = emitUnsignedOperand(value);
      } else {
        expression = emitExpression(value);
      }
      if (failed(expression)) {
        return failure();
      }
      *output << scalar.name << " = " << *expression << ";\n";
      ++scalarIndex;
    }
    return success();
  }

  [[nodiscard]] FailureOr<std::string>
  emitQuantizedGateParameter(const Value parameter,
                             const uint32_t precisionBits) {
    if (const auto quantized = mqt::angle::matchQuantizedRadians(parameter)) {
      if (quantized->bitWidth != precisionBits) {
        return failExpression(parameter,
                              "gate-angle precision metadata does not match "
                              "the canonical parameter bridge");
      }
      return emitAngleUse(quantized->bits, precisionBits);
    }
    return failExpression(
        parameter,
        "final gate-angle quantization does not match its parameter");
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
    for (const auto parameter : unitary.getParameters()) {
      auto expression =
          finalGatePrecision
              ? emitQuantizedGateParameter(parameter, *finalGatePrecision)
              : emitExpression(parameter);
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
      if (auto assertion = dyn_cast<cf::AssertOp>(&operation);
          assertion && isUnsignedDivisionSafetyAssert(assertion)) {
        continue;
      }
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
    if (body.getNumArguments() == 0) {
      fail(modifier, "multi-operation modifiers require a target qubit");
      return failure();
    }
    const auto helperName = uniqueName("gate", nextHelper);

    SmallVector<Value> captures;
    DenseSet<Value> captured;
    DenseMap<Value, uint32_t> capturedAngles;
    Value capturedQubit;
    const auto addCapture = [&](const Value value,
                                const std::optional<uint32_t> angle = {}) {
      if (captured.insert(value).second) {
        captures.push_back(value);
      }
      if (angle) {
        capturedAngles[value] = *angle;
      }
    };
    modifier.getRegion().walk([&](Operation* operation) {
      for (const auto result : operation->getResults()) {
        if (const auto angle = mqt::angle::matchQuantizedRadians(result);
            angle) {
          Value bits = angle->bits;
          if (!bits.getDefiningOp<arith::ConstantOp>() &&
              !modifier.getRegion().isAncestor(bits.getParentRegion())) {
            addCapture(bits, angle->bitWidth);
          }
        }
      }
      if (canonicalAngleOperations.contains(operation) ||
          isCanonicalAngleBridgeMember(*operation) ||
          (isa<cf::AssertOp>(operation) &&
           isUnsignedDivisionSafetyAssert(cast<cf::AssertOp>(operation)))) {
        return;
      }
      for (auto operand : operation->getOperands()) {
        if (modifier.getRegion().isAncestor(operand.getParentRegion())) {
          continue;
        }
        if (isa_and_nonnull<arith::ConstantOp>(operand.getDefiningOp())) {
          continue;
        }
        if (isa<QubitType>(operand.getType())) {
          capturedQubit = operand;
        } else {
          addCapture(operand, angleWidth(operand));
        }
      }
    });
    if (capturedQubit) {
      fail(modifier,
           "multi-operation modifier bodies cannot capture extra qubits");
      return failure();
    }
    if (llvm::any_of(captures, [&](const Value capture) {
          return !capturedAngles.contains(capture);
        })) {
      fail(modifier,
           "multi-operation modifier bodies cannot capture non-angle scalar "
           "values");
      return failure();
    }

    GateCall helperCall;
    helperCall.symbol = helperName;
    for (const auto capture : captures) {
      auto expression = emitAngleUse(capture, capturedAngles.at(capture));
      if (failed(expression)) {
        return failure();
      }
      helperCall.parameters.push_back(std::move(*expression));
    }

    DenseMap<Value, std::string> savedNames;
    DenseMap<Value, uint32_t> savedAngleWidths;
    auto saveAndMap = [&](const Value value, std::string name) {
      if (const auto found = valueNames.find(value);
          found != valueNames.end()) {
        savedNames.try_emplace(value, found->second);
      }
      valueNames[value] = std::move(name);
      if (const auto angle = capturedAngles.find(value);
          angle != capturedAngles.end()) {
        if (const auto previous = angleValues.find(value);
            previous != angleValues.end()) {
          savedAngleWidths[value] = previous->second;
        }
        angleValues[value] = angle->second;
      }
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
      if (const auto found = savedAngleWidths.find(value);
          found != savedAngleWidths.end()) {
        angleValues[value] = found->second;
      } else if (capturedAngles.contains(value)) {
        angleValues.erase(value);
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
