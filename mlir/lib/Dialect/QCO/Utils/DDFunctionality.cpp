/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Utils/DDFunctionality.h"

#include "dd/CachedEdge.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/Utils/ConstantFolding.h"
#include "mlir/Dialect/MQT/Utils/Modifiers.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mlir::qco {
namespace {

constexpr size_t MAX_CONTROL_FLOW_STEPS = 10'000;

struct QubitMap {
  DenseMap<Value, dd::Qubit> qubits;
  size_t numQubits = 0;

  void bind(Value value, dd::Qubit q) { qubits[value] = q; }

  [[nodiscard]] std::optional<dd::Qubit> lookup(Value value) const {
    const auto it = qubits.find(value);
    if (it == qubits.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  LogicalResult remapUnitary(UnitaryOpInterface unitary) {
    for (auto [in, out] :
         llvm::zip_equal(unitary.getInputQubits(), unitary.getOutputQubits())) {
      const auto q = lookup(in);
      if (!q) {
        return unitary.emitError()
               << "qubit SSA value is not mapped for QCO DD construction";
      }
      bind(out, *q);
    }
    return success();
  }

  FailureOr<SmallVector<dd::Qubit>> lookupRange(ValueRange values,
                                                Operation* op) const {
    SmallVector<dd::Qubit> out;
    out.reserve(values.size());
    for (Value value : values) {
      const auto q = lookup(value);
      if (!q) {
        return op->emitError()
               << "qubit SSA value is not mapped for QCO DD construction";
      }
      out.push_back(*q);
    }
    return out;
  }
};

/// Physical wires stored at each tensor index; extracted positions are empty.
using TensorSlots = SmallVector<std::optional<dd::Qubit>>;
using TensorState = std::shared_ptr<TensorSlots>;

struct TensorMap {
  DenseMap<Value, TensorState> tensors;

  void bind(Value value, TensorState slots) {
    tensors[value] = std::move(slots);
  }

  [[nodiscard]] TensorState lookup(Value value) const {
    const auto it = tensors.find(value);
    return it == tensors.end() ? nullptr : it->second;
  }

  [[nodiscard]] TensorMap clone() const {
    TensorMap copy;
    for (const auto& [value, slots] : tensors) {
      copy.bind(value, std::make_shared<TensorSlots>(*slots));
    }
    return copy;
  }
};

struct ClassicalEnv {
  struct RegisterBit {
    std::optional<bool> value;
    std::optional<dd::Qubit> deferredWire;
  };
  using RegisterState = std::vector<RegisterBit>;
  using MemRefState = SmallVector<Attribute>;

  DenseMap<Value, Attribute> values;
  DenseMap<Value, dd::Qubit> deferredMeasurements;
  Operation** deferredMeasurementUse = nullptr;
  /// Shared storage preserves CBit register identity across `func.call`.
  DenseMap<Value, std::shared_ptr<RegisterState>> registers;
  /// Shared storage preserves caller-visible writes through `func.call`.
  DenseMap<Value, std::shared_ptr<MemRefState>> memrefs;

  LogicalResult bindFrom(Value source, Value dest, Operation* op) {
    const auto it = values.find(source);
    if (it == values.end()) {
      if (deferredMeasurements.contains(source) &&
          deferredMeasurementUse != nullptr) {
        *deferredMeasurementUse = op;
        return failure();
      }
      return op->emitError()
             << "classical SSA value is not mapped for QCO DD simulation";
    }
    values[dest] = it->second;
    return success();
  }
};

struct DecodedGate {
  dd::GateType type = dd::GateType::None;
  std::vector<dd::fp> params;
};

struct WalkState {
  QubitMap* qubits;
  TensorMap* tensors;
  ClassicalEnv* classical;
  dd::Package* dd;
  std::mt19937_64* rng = nullptr;
  const DenseSet<Operation*>* deferredMeasurements = nullptr;
  DenseSet<dd::Qubit>* deferredMeasuredWires = nullptr;
  size_t remainingExecutionSteps = MAX_CONTROL_FLOW_STEPS;
  DenseSet<Operation*> activeCalls;
};

using RuntimeValue = std::variant<dd::Qubit, TensorState, Attribute,
                                  std::shared_ptr<ClassicalEnv::RegisterState>,
                                  std::shared_ptr<ClassicalEnv::MemRefState>>;
struct LoopRange {
  llvm::APInt induction, step;
  size_t trips;
};
struct SamplingPlan {
  bool dynamic = false;
  SmallVector<Value> outputs;
  DenseSet<Operation*> deferredMeasurements;
};
} // namespace

static LogicalResult consumeExecutionStep(WalkState& walk, Operation* op) {
  if (walk.remainingExecutionSteps == 0) {
    return op->emitError(
        "QCO DD execution exceeds the limit of 10000 control-flow steps");
  }
  --walk.remainingExecutionSteps;
  return success();
}

[[nodiscard]] static bool isQTensorType(Type type) {
  const auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getRank() == 1 &&
         isa<QubitType>(tensorType.getElementType());
}

static FailureOr<Attribute>
lookupAttribute(Value value, const ClassicalEnv& classical, Operation* op) {
  const auto it = classical.values.find(value);
  if (it == classical.values.end()) {
    if (classical.deferredMeasurements.contains(value) &&
        classical.deferredMeasurementUse != nullptr) {
      *classical.deferredMeasurementUse = op;
      return failure();
    }
    return op->emitError()
           << "classical SSA value is not mapped for QCO DD simulation";
  }
  return it->second;
}

static FailureOr<double>
resolveDouble(Value value, const ClassicalEnv& classical, Operation* op) {
  if (const auto it = classical.values.find(value);
      it != classical.values.end()) {
    if (const auto floating = dyn_cast<FloatAttr>(it->second)) {
      return floating.getValue().convertToDouble();
    }
  }
  if (const auto constant = mqt::valueToDouble(value)) {
    return *constant;
  }
  return op->emitError()
         << "floating-point SSA value has no concrete QCO DD binding";
}

/// `std::nullopt` if @p unitary is not a standard gate; failure if its unitary
/// parameters are not concrete.
static FailureOr<std::optional<DecodedGate>>
decodeStandardGate(UnitaryOpInterface unitary, const ClassicalEnv& classical) {
  Operation* op = unitary.getOperation();
  TypeSwitch<Operation*, dd::GateType> typeSwitch(op);
#define MQT_GATE(KEY, NAME, OP, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)   \
  typeSwitch.Case<KEY##Op>([](auto) { return dd::GateType::OP; });
#include "mlir/Conversion/GateTable.def"
  const auto type = typeSwitch.Default(dd::GateType::None);
  if (type == dd::GateType::None) {
    return std::optional<DecodedGate>{std::nullopt};
  }
  DecodedGate decoded{.type = type, .params = {}};
  for (Value param : unitary.getParameters()) {
    auto concrete = resolveDouble(param, classical, op);
    if (failed(concrete)) {
      return failure();
    }
    if (!std::isfinite(*concrete)) {
      return op->emitError()
             << "gate parameters must be finite for QCO DD simulation";
    }
    decoded.params.push_back(static_cast<dd::fp>(*concrete));
  }
  return std::optional{std::move(decoded)};
}

static dd::mCachedEdge
buildEmbeddedLocalDD(dd::Package& dd, const DynamicMatrix& local,
                     const DenseMap<dd::Qubit, size_t>& operandForWire,
                     size_t numOperands, int64_t level, size_t row,
                     size_t col) {
  if (level < 0) {
    return dd::mCachedEdge::terminal(
        local(static_cast<int64_t>(row), static_cast<int64_t>(col)));
  }
  const auto wire = static_cast<dd::Qubit>(level);
  const auto operand = operandForWire.find(wire);
  if (operand == operandForWire.end()) {
    const auto child = buildEmbeddedLocalDD(dd, local, operandForWire,
                                            numOperands, level - 1, row, col);
    return dd.makeDDNode<dd::mNode, dd::CachedEdge>(
        wire, {child, dd::mCachedEdge::zero(), dd::mCachedEdge::zero(), child});
  }

  const size_t operandMask = size_t{1} << (numOperands - 1 - operand->second);
  const auto edge00 = buildEmbeddedLocalDD(dd, local, operandForWire,
                                           numOperands, level - 1, row, col);
  const auto edge01 =
      buildEmbeddedLocalDD(dd, local, operandForWire, numOperands, level - 1,
                           row, col | operandMask);
  const auto edge10 =
      buildEmbeddedLocalDD(dd, local, operandForWire, numOperands, level - 1,
                           row | operandMask, col);
  const auto edge11 =
      buildEmbeddedLocalDD(dd, local, operandForWire, numOperands, level - 1,
                           row | operandMask, col | operandMask);
  return dd.makeDDNode<dd::mNode, dd::CachedEdge>(
      wire, {edge00, edge01, edge10, edge11});
}

static dd::MatrixDD makeEmbeddedLocalDD(dd::Package& dd,
                                        const DynamicMatrix& local,
                                        size_t numQubits,
                                        ArrayRef<dd::Qubit> wires) {
  DenseMap<dd::Qubit, size_t> operandForWire;
  for (auto [operand, wire] : llvm::enumerate(wires)) {
    operandForWire[wire] = operand;
  }
  const auto root =
      buildEmbeddedLocalDD(dd, local, operandForWire, wires.size(),
                           static_cast<int64_t>(numQubits) - 1, 0, 0);
  return {.p = root.p, .w = dd.cn.lookup(root.w)};
}

template <typename StateDD>
static LogicalResult applyUnitaryMatrix(UnitaryOpInterface unitary,
                                        WalkState& walk, StateDD& state) {
  Operation* op = unitary.getOperation();
  if (auto gphase = dyn_cast<GPhaseOp>(op)) {
    auto theta = resolveDouble(gphase.getTheta(), *walk.classical, op);
    if (failed(theta)) {
      return failure();
    }
    if (!std::isfinite(*theta)) {
      return gphase.emitError()
             << "global phase must be finite for QCO DD simulation";
    }
    auto id = dd::Package::makeIdent();
    id.w = walk.dd->cn.lookup(std::cos(*theta), std::sin(*theta));
    state = walk.dd->applyOperation(id, state);
    return success();
  }
  if (isa<BarrierOp>(op)) {
    return walk.qubits->remapUnitary(unitary);
  }
  if (!unitary.hasCompileTimeKnownUnitaryMatrix()) {
    return unitary.emitError()
           << "unitary must have a compile-time constant matrix";
  }

  DynamicMatrix local;
  if (!unitary.getUnitaryMatrixDynamic(local)) {
    return unitary.emitError()
           << "unitary must have a compile-time constant matrix";
  }

  auto wiresOr = walk.qubits->lookupRange(unitary.getInputQubits(), op);
  if (failed(wiresOr)) {
    return failure();
  }
  ArrayRef<dd::Qubit> wires = *wiresOr;
  if (wires.size() >= 63 ||
      local.rows() != static_cast<int64_t>(size_t{1} << wires.size())) {
    return unitary.emitError()
           << "unitary matrix dimension does not match its target count";
  }

  if (wires.size() == 1) {
    const dd::GateMatrix mat{local(0, 0), local(0, 1), local(1, 0),
                             local(1, 1)};
    state = walk.dd->applyOperation(walk.dd->makeGateDD(mat, wires[0]), state);
    return walk.qubits->remapUnitary(unitary);
  }

  if (wires.size() == 2) {
    dd::TwoQubitGateMatrix mat{};
    for (size_t row = 0; row < mat.size(); ++row) {
      for (size_t col = 0; col < mat[row].size(); ++col) {
        mat[row][col] =
            local(static_cast<int64_t>(row), static_cast<int64_t>(col));
      }
    }
    state = walk.dd->applyOperation(
        walk.dd->makeTwoQubitGateDD(mat, wires[0], wires[1]), state);
    return walk.qubits->remapUnitary(unitary);
  }

  if (wires.size() == 3) {
    dd::ThreeQubitGateMatrix mat{};
    for (size_t row = 0; row < mat.size(); ++row) {
      for (size_t col = 0; col < mat[row].size(); ++col) {
        mat[row][col] =
            local(static_cast<int64_t>(row), static_cast<int64_t>(col));
      }
    }
    state = walk.dd->applyOperation(
        walk.dd->makeThreeQubitGateDD(mat, wires[0], wires[1], wires[2]),
        state);
    return walk.qubits->remapUnitary(unitary);
  }

  state = walk.dd->applyOperation(
      makeEmbeddedLocalDD(*walk.dd, local, walk.qubits->numQubits, wires),
      state);
  return walk.qubits->remapUnitary(unitary);
}

template <typename StateDD>
static LogicalResult applyDecodedStandard(UnitaryOpInterface unitary,
                                          const DecodedGate& gate,
                                          const dd::Controls& controls,
                                          WalkState& walk, StateDD& state) {
  SmallVector<Value> targetVals;
  for (size_t i = 0; i < unitary.getNumTargets(); ++i) {
    targetVals.push_back(unitary.getInputTarget(i));
  }
  auto targets = walk.qubits->lookupRange(targetVals, unitary.getOperation());
  if (failed(targets)) {
    return failure();
  }
  state = walk.dd->applyOperation(
      dd::getGateDD(*walk.dd, gate.type, gate.params, controls,
                    {targets->begin(), targets->end()}),
      state);
  return walk.qubits->remapUnitary(unitary);
}

static LogicalResult validateReturn(func::ReturnOp returnOp,
                                    const QubitMap& qubits,
                                    const TensorMap& tensors) {
  dd::Qubit expected = 0;
  for (Value value : returnOp.getOperands()) {
    if (isQTensorType(value.getType())) {
      const auto slots = tensors.lookup(value);
      if (!slots) {
        return returnOp.emitError()
               << "returned qtensor is not mapped for QCO DD simulation";
      }
      for (const auto wire : *slots) {
        if (!wire || *wire != expected) {
          return returnOp.emitError()
                 << "returned qubits must preserve canonical wire order";
        }
        ++expected;
      }
      continue;
    }
    if (!isa<QubitType>(value.getType())) {
      continue;
    }
    const auto mapped = qubits.lookup(value);
    if (!mapped) {
      return returnOp.emitError()
             << "returned qubit SSA value is not mapped for QCO DD "
                "construction";
    }
    if (*mapped != expected) {
      return returnOp.emitError()
             << "returned qubits must preserve canonical wire order; qubit "
                "result "
             << static_cast<size_t>(expected) << " maps to wire "
             << static_cast<size_t>(*mapped);
    }
    ++expected;
  }
  return success();
}

[[nodiscard]] static bool isSupportedClassicalType(Type type) {
  return isa<IndexType, IntegerType>(type) || type.isF64();
}

static LogicalResult recordConstant(arith::ConstantOp constant,
                                    ClassicalEnv& classical) {
  if (!isSupportedClassicalType(constant.getType())) {
    return constant.emitError()
           << "QCO DD simulation only supports integer, index, and f64 values";
  }
  classical.values[constant.getResult()] = constant.getValue();
  return success();
}

static LogicalResult
applyArgumentBindings(func::FuncOp func,
                      const DDArgumentBindings& argumentBindings,
                      ClassicalEnv& classical) {
  size_t boundArguments = 0;
  for (Value argument : func.getArguments()) {
    const auto binding = argumentBindings.find(argument);
    if (binding == argumentBindings.end()) {
      continue;
    }
    ++boundArguments;
    const Attribute attr = binding->second;
    const Type type = argument.getType();
    if (isQTensorType(type)) {
      const auto extent = dyn_cast<IntegerAttr>(attr);
      if (cast<RankedTensorType>(type).isDynamicDim(0) && extent &&
          isa<IndexType>(extent.getType())) {
        continue;
      }
    } else if (const auto typed = dyn_cast<TypedAttr>(attr);
               isSupportedClassicalType(type) && typed &&
               typed.getType() == type) {
      classical.values[argument] = attr;
      continue;
    }
    return func.emitError() << "QCO DD binding attribute " << attr
                            << " does not match argument type " << type;
  }
  if (boundArguments != argumentBindings.size()) {
    return func.emitError()
           << "QCO DD bindings must target entry-block arguments";
  }
  return success();
}

static FailureOr<bool> lookupBool(Value value, const ClassicalEnv& classical,
                                  Operation* op) {
  auto attr = lookupAttribute(value, classical, op);
  const auto integer =
      succeeded(attr) ? dyn_cast<IntegerAttr>(*attr) : IntegerAttr{};
  if (!integer || !value.getType().isInteger(1)) {
    return failure();
  }
  return !integer.getValue().isZero();
}

static FailureOr<int64_t>
lookupIndex(Value value, const ClassicalEnv& classical, Operation* op) {
  auto attr = lookupAttribute(value, classical, op);
  const auto integer =
      succeeded(attr) ? dyn_cast<IntegerAttr>(*attr) : IntegerAttr{};
  if (!integer || !isa<IndexType>(value.getType())) {
    return failure();
  }
  return integer.getValue().getSExtValue();
}

static FailureOr<double> lookupFloat(Value value, const ClassicalEnv& classical,
                                     Operation* op) {
  auto attr = lookupAttribute(value, classical, op);
  const auto floating =
      succeeded(attr) ? dyn_cast<FloatAttr>(*attr) : FloatAttr{};
  if (!floating || !value.getType().isF64()) {
    return failure();
  }
  return floating.getValue().convertToDouble();
}

static FailureOr<llvm::APInt>
lookupInteger(Value value, const ClassicalEnv& classical, Operation* op) {
  if (!isa<IntegerType, IndexType>(value.getType())) {
    return op->emitError() << "expected an integer or index SSA value";
  }
  auto attr = lookupAttribute(value, classical, op);
  const auto integer =
      succeeded(attr) ? dyn_cast<IntegerAttr>(*attr) : IntegerAttr{};
  if (!integer) {
    return failure();
  }
  return integer.getValue();
}

static LogicalResult bindInteger(Value dest, const llvm::APInt& value,
                                 ClassicalEnv& classical) {
  const Type type = dest.getType();
  if (!isa<IntegerType, IndexType>(type)) {
    return failure();
  }
  const unsigned width =
      isa<IndexType>(type) ? 64U : cast<IntegerType>(type).getWidth();
  classical.values[dest] = IntegerAttr::get(type, value.zextOrTrunc(width));
  return success();
}

static LogicalResult allocateRegister(cbit::AllocOp alloc,
                                      ClassicalEnv& classical) {
  const auto width =
      static_cast<size_t>(alloc.getResult().getType().getWidth());
  ClassicalEnv::RegisterBit initialValue;
  if (alloc.getInitialization() == cbit::Initialization::Zero) {
    initialValue.value = false;
  }
  classical.registers[alloc.getResult()] =
      std::make_shared<ClassicalEnv::RegisterState>(width, initialValue);
  return success();
}

static FailureOr<size_t> resolveRegisterIndex(Value index,
                                              cbit::RegisterType type,
                                              const ClassicalEnv& classical,
                                              Operation* op) {
  auto resolved = lookupIndex(index, classical, op);
  if (failed(resolved)) {
    return failure();
  }
  if (*resolved < 0 || *resolved >= type.getWidth()) {
    return op->emitError() << "CBit register index " << *resolved
                           << " is out of bounds for width " << type.getWidth();
  }
  return static_cast<size_t>(*resolved);
}

static LogicalResult storeRegister(cbit::StoreOp store,
                                   ClassicalEnv& classical) {
  const auto regIt = classical.registers.find(store.getReg());
  if (regIt == classical.registers.end()) {
    return store.emitError()
           << "CBit register is not mapped for QCO DD simulation";
  }
  auto index = resolveRegisterIndex(store.getIndex(), store.getReg().getType(),
                                    classical, store);
  if (failed(index)) {
    return failure();
  }
  auto& cell = (*regIt->second)[*index];
  if (const auto deferred =
          classical.deferredMeasurements.find(store.getValue());
      deferred != classical.deferredMeasurements.end()) {
    cell.value.reset();
    cell.deferredWire = deferred->second;
    return success();
  }
  auto value = lookupBool(store.getValue(), classical, store);
  if (failed(value)) {
    return failure();
  }
  cell.value.emplace(*value);
  cell.deferredWire.reset();
  return success();
}

static LogicalResult loadRegister(cbit::LoadOp load, ClassicalEnv& classical) {
  const auto regIt = classical.registers.find(load.getReg());
  if (regIt == classical.registers.end()) {
    return load.emitError()
           << "CBit register is not mapped for QCO DD simulation";
  }
  auto index = resolveRegisterIndex(load.getIndex(), load.getReg().getType(),
                                    classical, load);
  if (failed(index)) {
    return failure();
  }
  const auto& cell = (*regIt->second)[*index];
  if (cell.deferredWire && classical.deferredMeasurementUse != nullptr) {
    *classical.deferredMeasurementUse = load.getOperation();
    return failure();
  }
  if (!cell.value) {
    return load.emitError() << "read from an undefined CBit register element";
  }
  return bindInteger(load.getResult(),
                     llvm::APInt(1, static_cast<uint64_t>(*cell.value)),
                     classical);
}

static FailureOr<Attribute*> lookupMemRefSlot(Value memref, ValueRange indices,
                                              ClassicalEnv& classical,
                                              Operation* op) {
  const auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type || type.getRank() != 1 || indices.size() != 1 ||
      !isSupportedClassicalType(type.getElementType())) {
    return op->emitError()
           << "QCO DD simulation only supports one-dimensional memrefs of "
              "integer, index, or f64 values";
  }
  auto index = lookupIndex(indices[0], classical, op);
  if (failed(index)) {
    return failure();
  }
  const auto it = classical.memrefs.find(memref);
  if (it == classical.memrefs.end()) {
    return op->emitError()
           << "classical memref is not mapped for QCO DD simulation";
  }
  if (*index < 0 || static_cast<size_t>(*index) >= it->second->size()) {
    return op->emitError()
           << "classical memref index out of range for QCO DD simulation";
  }
  return &(*it->second)[static_cast<size_t>(*index)];
}

static LogicalResult applyMemRefAlloc(memref::AllocOp alloc,
                                      ClassicalEnv& classical) {
  const auto type = dyn_cast<MemRefType>(alloc.getType());
  if (!type || type.getRank() != 1 ||
      !isSupportedClassicalType(type.getElementType())) {
    return alloc.emitError()
           << "QCO DD simulation only supports one-dimensional memrefs of "
              "integer, index, or f64 values";
  }
  if (!alloc.getSymbolOperands().empty()) {
    return alloc.emitError()
           << "QCO DD simulation does not support symbolic memref operands";
  }
  int64_t size = type.getDimSize(0);
  if (type.isDynamicDim(0)) {
    auto dynamicSize =
        lookupIndex(alloc.getDynamicSizes()[0], classical, alloc);
    if (failed(dynamicSize)) {
      return failure();
    }
    size = *dynamicSize;
  }
  if (size < 0) {
    return alloc.emitError() << "classical memref size must be non-negative";
  }
  classical.memrefs[alloc.getResult()] =
      std::make_shared<ClassicalEnv::MemRefState>(static_cast<size_t>(size));
  return success();
}

static LogicalResult applyMemRefStore(memref::StoreOp store,
                                      ClassicalEnv& classical) {
  auto slot =
      lookupMemRefSlot(store.getMemref(), store.getIndices(), classical, store);
  if (failed(slot)) {
    return failure();
  }
  auto value = lookupAttribute(store.getValue(), classical, store);
  if (failed(value)) {
    return failure();
  }
  **slot = *value;
  return success();
}

static LogicalResult applyMemRefLoad(memref::LoadOp load,
                                     ClassicalEnv& classical) {
  auto slot =
      lookupMemRefSlot(load.getMemref(), load.getIndices(), classical, load);
  if (failed(slot)) {
    return failure();
  }
  if (!**slot) {
    return load.emitError()
           << "read from an uninitialized classical memref element";
  }
  classical.values[load.getResult()] = **slot;
  return success();
}

template <typename OpTy, typename Combine>
static LogicalResult applyDivision(OpTy op, ClassicalEnv& classical,
                                   Combine combine) {
  auto rhs = lookupInteger(op.getRhs(), classical, op);
  if (failed(rhs)) {
    return failure();
  }
  if (rhs->isZero()) {
    return op.emitError() << "division by zero during QCO DD simulation";
  }
  auto lhs = lookupInteger(op.getLhs(), classical, op);
  if (failed(lhs)) {
    return failure();
  }
  return bindInteger(op.getResult(), combine(*lhs, *rhs), classical);
}

static LogicalResult applyIntegerCast(Value in, Value out, Operation* op,
                                      ClassicalEnv& classical, bool isSigned) {
  auto value = lookupInteger(in, classical, op);
  if (failed(value)) {
    return failure();
  }
  const unsigned width = isa<IndexType>(out.getType())
                             ? 64U
                             : cast<IntegerType>(out.getType()).getWidth();
  if (width > value->getBitWidth()) {
    *value = isSigned ? value->sext(width) : value->zext(width);
  } else if (width < value->getBitWidth()) {
    *value = value->trunc(width);
  }
  return bindInteger(out, *value, classical);
}

static LogicalResult foldClassicalOp(Operation& op, ClassicalEnv& classical) {
  if (llvm::any_of(op.getOperandTypes(),
                   [](Type type) { return !isSupportedClassicalType(type); }) ||
      llvm::any_of(op.getResultTypes(),
                   [](Type type) { return !isSupportedClassicalType(type); })) {
    return op.emitError()
           << "QCO DD simulation only supports integer, index, and f64 values";
  }

  // Fold a clone because some arithmetic folders canonicalize in place.
  Operation* clone = op.clone();
  const auto destroyClone = llvm::make_scope_exit([&] { clone->destroy(); });
  Attribute result;
  for (unsigned attempt = 0; attempt < 2 && !result; ++attempt) {
    SmallVector<Attribute> operands;
    operands.reserve(clone->getNumOperands());
    for (Value operand : clone->getOperands()) {
      auto attr = lookupAttribute(operand, classical, &op);
      if (failed(attr)) {
        return failure();
      }
      operands.push_back(*attr);
    }

    if (isa<arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp>(op)) {
      const auto lhs = cast<IntegerAttr>(operands[0]);
      const auto rhs = cast<IntegerAttr>(operands[1]);
      if (rhs.getValue().uge(lhs.getValue().getBitWidth())) {
        return op.emitError()
               << "shift amount out of range for QCO DD simulation";
      }
    }

    SmallVector<OpFoldResult, 1> results;
    if (failed(clone->fold(operands, results))) {
      break;
    }
    if (results.size() == 1) {
      result = dyn_cast_if_present<Attribute>(results.front());
      if (!result) {
        const auto value = cast<Value>(results.front());
        if (value != clone->getResult(0)) {
          return classical.bindFrom(value, op.getResult(0), &op);
        }
      }
    } else if (!results.empty()) {
      break;
    }
  }
  if (!result) {
    return op.emitError()
           << "could not evaluate classical op during QCO DD simulation";
  }
  classical.values[op.getResult(0)] = result;
  return success();
}

static LogicalResult applyClassicalOp(Operation& op, ClassicalEnv& classical) {
  const auto isUnsupportedFloat = [](Type type) {
    return isa<FloatType>(type) && !type.isF64();
  };
  if (llvm::any_of(op.getOperandTypes(), isUnsupportedFloat) ||
      llvm::any_of(op.getResultTypes(), isUnsupportedFloat)) {
    return op.emitError()
           << "QCO DD simulation only supports f64 classical values";
  }
  return TypeSwitch<Operation*, LogicalResult>(&op)
      .Case<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::AddIOp,
            arith::SubIOp, arith::MulIOp, arith::ShLIOp, arith::ShRUIOp,
            arith::ShRSIOp, arith::CmpIOp, arith::AddFOp, arith::SubFOp,
            arith::MulFOp, arith::DivFOp, arith::RemFOp, arith::NegFOp,
            arith::CmpFOp, arith::SIToFPOp, arith::UIToFPOp, arith::MaxSIOp,
            arith::MinSIOp, arith::MaxUIOp, arith::MinUIOp, arith::MaximumFOp,
            arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp, math::AbsFOp,
            math::CeilOp, math::CosOp, math::ExpOp, math::FloorOp, math::LogOp,
            math::SinOp, math::SqrtOp, math::TanOp, math::PowFOp>(
          [&](Operation* foldable) {
            return foldClassicalOp(*foldable, classical);
          })
      .Case<arith::DivUIOp>([&](arith::DivUIOp value) {
        return applyDivision(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs.udiv(rhs);
            });
      })
      .Case<arith::DivSIOp>([&](arith::DivSIOp value) {
        return applyDivision(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs.sdiv(rhs);
            });
      })
      .Case<arith::RemUIOp>([&](arith::RemUIOp value) {
        return applyDivision(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs.urem(rhs);
            });
      })
      .Case<arith::RemSIOp>([&](arith::RemSIOp value) {
        return applyDivision(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs.srem(rhs);
            });
      })
      .Case<arith::SelectOp>([&](arith::SelectOp select) -> LogicalResult {
        auto condition = lookupBool(select.getCondition(), classical, select);
        if (failed(condition)) {
          return failure();
        }
        Value selected =
            *condition ? select.getTrueValue() : select.getFalseValue();
        return classical.bindFrom(selected, select.getResult(), select);
      })
      .Case<arith::ExtUIOp>([&](arith::ExtUIOp ext) {
        return applyIntegerCast(ext.getIn(), ext.getOut(), ext, classical,
                                false);
      })
      .Case<arith::ExtSIOp>([&](arith::ExtSIOp cast) {
        return applyIntegerCast(cast.getIn(), cast.getOut(), cast, classical,
                                true);
      })
      .Case<arith::IndexCastUIOp>([&](arith::IndexCastUIOp cast) {
        return applyIntegerCast(cast.getIn(), cast.getOut(), cast, classical,
                                false);
      })
      .Case<arith::IndexCastOp>([&](arith::IndexCastOp cast) {
        return applyIntegerCast(cast.getIn(), cast.getOut(), cast, classical,
                                true);
      })
      .Case<arith::TruncIOp>([&](arith::TruncIOp cast) {
        return applyIntegerCast(cast.getIn(), cast.getOut(), cast, classical,
                                false);
      })
      .Case<arith::FPToSIOp, arith::FPToUIOp>(
          [&](Operation* castOp) -> LogicalResult {
            auto value = lookupFloat(castOp->getOperand(0), classical, castOp);
            if (failed(value)) {
              return failure();
            }
            Value out = castOp->getResult(0);
            const unsigned width = cast<IntegerType>(out.getType()).getWidth();
            const bool isSigned = isa<arith::FPToSIOp>(castOp);
            llvm::APSInt result(width, /*isUnsigned=*/!isSigned);
            bool exact = false;
            const auto status = llvm::APFloat(*value).convertToInteger(
                result, llvm::APFloat::rmTowardZero, &exact);
            if ((status & llvm::APFloat::opInvalidOp) != 0) {
              return castOp->emitError()
                     << "floating-point value is outside the destination "
                        "integer range during QCO DD simulation";
            }
            return bindInteger(out, result, classical);
          })
      .Default([](Operation* unsupported) {
        return unsupported->emitError()
               << "unsupported classical op for QCO DD simulation: "
               << unsupported->getName().getStringRef();
      });
}

static FailureOr<LoopRange> resolveLoop(scf::ForOp forOp,
                                        ClassicalEnv& classical) {
  auto lower = lookupInteger(forOp.getLowerBound(), classical, forOp);
  auto upper = lookupInteger(forOp.getUpperBound(), classical, forOp);
  auto step = lookupInteger(forOp.getStep(), classical, forOp);
  if (failed(lower) || failed(upper) || failed(step)) {
    return failure();
  }
  if (!step->isStrictlyPositive()) {
    return forOp.emitError(
        "scf.for step must be positive for QCO DD simulation");
  }

  const bool unsignedCmp = forOp.getUnsignedCmp();
  if (!(unsignedCmp ? lower->ult(*upper) : lower->slt(*upper))) {
    return LoopRange{.induction = *lower, .step = *step, .trips = 0};
  }

  const unsigned wideWidth = lower->getBitWidth() + 1;
  const auto extend = [unsignedCmp, wideWidth](const llvm::APInt& value) {
    return unsignedCmp ? value.zext(wideWidth) : value.sext(wideWidth);
  };
  const llvm::APInt lowerWide = extend(*lower);
  const llvm::APInt upperWide = extend(*upper);
  const llvm::APInt stepWide = step->zext(wideWidth);
  const llvm::APInt span = upperWide - lowerWide;
  const llvm::APInt trips =
      (span + stepWide - llvm::APInt(wideWidth, 1)).udiv(stepWide);
  const size_t limited = trips.getLimitedValue(MAX_CONTROL_FLOW_STEPS + 1);
  return LoopRange{.induction = lowerWide, .step = stepWide, .trips = limited};
}

static LogicalResult bindValuePairs(ValueRange sources, ValueRange dests,
                                    WalkState& walk, Operation* op);

static LogicalResult bindValuePairs(ValueRange sources, ValueRange dests,
                                    WalkState& walk, Operation* op) {
  SmallVector<RuntimeValue> values;
  values.reserve(sources.size());
  for (auto [src, dest] : llvm::zip_equal(sources, dests)) {
    if (isa<QubitType>(dest.getType())) {
      const auto q = walk.qubits->lookup(src);
      if (!q) {
        return op->emitError()
               << "qubit SSA value is not mapped for QCO DD construction";
      }
      values.emplace_back(*q);
    } else if (isQTensorType(dest.getType())) {
      const auto slots = walk.tensors->lookup(src);
      if (!slots) {
        return op->emitError()
               << "qtensor SSA value is not mapped for QCO DD simulation";
      }
      values.emplace_back(slots);
    } else if (isa<cbit::RegisterType>(dest.getType())) {
      const auto it = walk.classical->registers.find(src);
      if (it == walk.classical->registers.end()) {
        return op->emitError()
               << "CBit register is not mapped for QCO DD simulation";
      }
      values.emplace_back(it->second);
    } else if (isa<MemRefType>(dest.getType())) {
      const auto it = walk.classical->memrefs.find(src);
      if (it == walk.classical->memrefs.end()) {
        return op->emitError()
               << "classical memref is not mapped for QCO DD simulation";
      }
      values.emplace_back(it->second);
    } else {
      if (const auto deferred = walk.classical->deferredMeasurements.find(src);
          deferred != walk.classical->deferredMeasurements.end()) {
        values.emplace_back(deferred->second);
        continue;
      }
      const auto value = walk.classical->values.find(src);
      if (value == walk.classical->values.end()) {
        return op->emitError()
               << "classical SSA value is not mapped for QCO DD simulation";
      }
      values.emplace_back(value->second);
    }
  }

  for (auto [value, dest] : llvm::zip_equal(values, dests)) {
    if (isa<QubitType>(dest.getType())) {
      walk.qubits->bind(dest, std::get<dd::Qubit>(value));
    } else if (isQTensorType(dest.getType())) {
      walk.tensors->bind(dest, std::get<TensorState>(value));
    } else if (isa<cbit::RegisterType>(dest.getType())) {
      walk.classical->registers[dest] =
          std::get<std::shared_ptr<ClassicalEnv::RegisterState>>(value);
    } else if (isa<MemRefType>(dest.getType())) {
      walk.classical->memrefs[dest] =
          std::get<std::shared_ptr<ClassicalEnv::MemRefState>>(value);
    } else if (std::holds_alternative<dd::Qubit>(value)) {
      walk.classical->values.erase(dest);
      walk.classical->deferredMeasurements[dest] = std::get<dd::Qubit>(value);
    } else {
      walk.classical->deferredMeasurements.erase(dest);
      walk.classical->values[dest] = std::get<Attribute>(value);
    }
  }
  return success();
}

static LogicalResult bindYieldResults(YieldOp yield,
                                      ValueRange classicalResults,
                                      ValueRange linearResults,
                                      WalkState& walk) {
  const size_t numClassical = classicalResults.size();
  if (failed(bindValuePairs(yield.getOperands().take_front(numClassical),
                            classicalResults, walk, yield))) {
    return failure();
  }
  return bindValuePairs(yield.getOperands().drop_front(numClassical),
                        linearResults, walk, yield);
}

template <typename StateDD>
static LogicalResult applyOp(Operation& op, WalkState& walk, StateDD& state);

template <typename StateDD>
static FailureOr<func::ReturnOp>
walkFunctionBody(func::FuncOp func, WalkState& walk, StateDD& state);

template <typename StateDD>
static LogicalResult walkBlock(Block& block, WalkState& walk, StateDD& state) {
  for (Operation& op : block.without_terminator()) {
    if (failed(applyOp(op, walk, state))) {
      return failure();
    }
  }
  return success();
}

template <typename StateDD>
static LogicalResult
applyRegionBranch(ValueRange linearOperands, Block& block,
                  ValueRange classicalResults, ValueRange linearResults,
                  WalkState& walk, StateDD& state, Operation* parent) {
  if (failed(
          bindValuePairs(linearOperands, block.getArguments(), walk, parent))) {
    return failure();
  }
  if (failed(consumeExecutionStep(walk, parent))) {
    return failure();
  }
  if (failed(walkBlock(block, walk, state))) {
    return failure();
  }
  return bindYieldResults(cast<YieldOp>(block.getTerminator()),
                          classicalResults, linearResults, walk);
}

template <typename StateDD>
static LogicalResult applyScfRegion(Region& region, ValueRange results,
                                    WalkState& walk, StateDD& state,
                                    Operation* parent) {
  if (failed(consumeExecutionStep(walk, parent))) {
    return failure();
  }
  Block& block = region.front();
  if (failed(walkBlock(block, walk, state))) {
    return failure();
  }
  auto yield = cast<scf::YieldOp>(block.getTerminator());
  return bindValuePairs(yield.getOperands(), results, walk, parent);
}

static FailureOr<TensorSlots> allocateZeroQubits(size_t count, WalkState& walk,
                                                 dd::VectorDD& state,
                                                 Operation* op) {
  if (count > dd::Package::MAX_POSSIBLE_QUBITS - walk.qubits->numQubits) {
    return op->emitError() << "QCO function exceeds the supported qubit range";
  }
  const size_t required = walk.qubits->numQubits + count;
  if (walk.dd->qubits() < required) {
    walk.dd->resize(required);
  }

  const size_t first = walk.qubits->numQubits;
  auto zeros = dd::makeZeroState(count, *walk.dd, first);
  auto extended = walk.dd->kronecker(zeros, state, first, /*incIdx=*/false);
  walk.dd->incRef(extended);
  walk.dd->decRef(zeros);
  walk.dd->decRef(state);
  state = extended;

  TensorSlots slots;
  slots.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    slots.emplace_back(static_cast<dd::Qubit>(first + i));
  }
  walk.qubits->numQubits += count;
  return slots;
}

static LogicalResult checkDeferredMeasurementUse(UnitaryOpInterface unitary,
                                                 WalkState& walk) {
  if (walk.deferredMeasuredWires == nullptr ||
      isa<BarrierOp>(unitary.getOperation())) {
    return success();
  }
  auto wires = walk.qubits->lookupRange(unitary.getInputQubits(),
                                        unitary.getOperation());
  if (failed(wires)) {
    return failure();
  }
  if (llvm::none_of(*wires, [&](dd::Qubit wire) {
        return walk.deferredMeasuredWires->contains(wire);
      })) {
    return success();
  }
  *walk.classical->deferredMeasurementUse = unitary.getOperation();
  return failure();
}

template <typename StateDD>
static LogicalResult applyOp(Operation& op, WalkState& walk, StateDD& state) {
  return TypeSwitch<Operation*, LogicalResult>(&op)
      .template Case<StaticOp, SinkOp, qtensor::DeallocOp>(
          [](auto) { return success(); })
      .template Case<arith::ConstantOp>([&](arith::ConstantOp constant) {
        return recordConstant(constant, *walk.classical);
      })
      .template Case<AllocOp>([&](AllocOp alloc) -> LogicalResult {
        if constexpr (!std::is_same_v<StateDD, dd::VectorDD>) {
          if (!walk.qubits->lookup(alloc.getResult())) {
            return alloc.emitError()
                   << "dynamic qubit allocation is not supported for QCO DD "
                      "functionality construction";
          }
          return success();
        } else {
          auto slots = allocateZeroQubits(1, walk, state, alloc);
          if (failed(slots)) {
            return failure();
          }
          walk.qubits->bind(alloc.getResult(), *slots->front());
          return success();
        }
      })
      .template Case<qtensor::AllocOp>(
          [&](qtensor::AllocOp alloc) -> LogicalResult {
            if constexpr (!std::is_same_v<StateDD, dd::VectorDD>) {
              return alloc.emitError()
                     << "qtensor allocation is not supported for QCO DD "
                        "functionality construction";
            } else {
              auto size = lookupIndex(alloc.getSize(), *walk.classical, alloc);
              if (failed(size)) {
                return failure();
              }
              if (*size <= 0) {
                return alloc.emitError()
                       << "qtensor allocation size must be positive";
              }
              auto slots = allocateZeroQubits(static_cast<size_t>(*size), walk,
                                              state, alloc);
              if (failed(slots)) {
                return failure();
              }
              walk.tensors->bind(
                  alloc.getResult(),
                  std::make_shared<TensorSlots>(std::move(*slots)));
              return success();
            }
          })
      .template Case<qtensor::FromElementsOp>(
          [&](qtensor::FromElementsOp fromElements) -> LogicalResult {
            auto wires = walk.qubits->lookupRange(fromElements.getElements(),
                                                  fromElements);
            if (failed(wires)) {
              return failure();
            }
            TensorSlots slots;
            slots.reserve(wires->size());
            for (const dd::Qubit wire : *wires) {
              slots.emplace_back(wire);
            }
            walk.tensors->bind(fromElements.getResult(),
                               std::make_shared<TensorSlots>(std::move(slots)));
            return success();
          })
      .template Case<qtensor::ExtractOp>(
          [&](qtensor::ExtractOp extract) -> LogicalResult {
            const auto input = walk.tensors->lookup(extract.getTensor());
            auto index =
                lookupIndex(extract.getIndex(), *walk.classical, extract);
            if (!input || failed(index)) {
              if (!input) {
                extract.emitError()
                    << "qtensor is not mapped for QCO DD simulation";
              }
              return failure();
            }
            if (*index < 0 || static_cast<size_t>(*index) >= input->size()) {
              return extract.emitError() << "qtensor index out of range";
            }
            auto& wire = (*input)[static_cast<size_t>(*index)];
            if (!wire) {
              return extract.emitError()
                     << "qtensor element has already been extracted";
            }
            walk.qubits->bind(extract.getResult(), *wire);
            wire.reset();
            walk.tensors->bind(extract.getOutTensor(), input);
            return success();
          })
      .template Case<qtensor::InsertOp>(
          [&](qtensor::InsertOp insert) -> LogicalResult {
            const auto input = walk.tensors->lookup(insert.getDest());
            const auto wire = walk.qubits->lookup(insert.getScalar());
            auto index =
                lookupIndex(insert.getIndex(), *walk.classical, insert);
            if (!input || !wire || failed(index)) {
              if (!input || !wire) {
                insert.emitError()
                    << "qtensor or qubit is not mapped for QCO DD simulation";
              }
              return failure();
            }
            if (*index < 0 || static_cast<size_t>(*index) >= input->size()) {
              return insert.emitError() << "qtensor index out of range";
            }
            (*input)[static_cast<size_t>(*index)] = wire;
            walk.tensors->bind(insert.getResult(), input);
            return success();
          })
      .template Case<memref::AllocOp>([&](memref::AllocOp alloc) {
        return applyMemRefAlloc(alloc, *walk.classical);
      })
      .template Case<memref::StoreOp>([&](memref::StoreOp store) {
        return applyMemRefStore(store, *walk.classical);
      })
      .template Case<memref::LoadOp>([&](memref::LoadOp load) {
        return applyMemRefLoad(load, *walk.classical);
      })
      .template Case<cbit::AllocOp>([&](cbit::AllocOp alloc) {
        return allocateRegister(alloc, *walk.classical);
      })
      .template Case<cbit::LoadOp>([&](cbit::LoadOp load) {
        return loadRegister(load, *walk.classical);
      })
      .template Case<cbit::StoreOp>([&](cbit::StoreOp store) {
        return storeRegister(store, *walk.classical);
      })
      .template Case<memref::DeallocOp>([](auto) { return success(); })
      .template Case<MeasureOp>([&](MeasureOp measureOp) -> LogicalResult {
        if constexpr (!std::is_same_v<StateDD, dd::VectorDD>) {
          return measureOp.emitError()
                 << "measurements are not supported for QCO DD functionality "
                    "construction";
        } else {
          const bool deferred =
              walk.deferredMeasurements != nullptr &&
              walk.deferredMeasurements->contains(measureOp.getOperation());
          if (walk.rng == nullptr && !deferred) {
            return measureOp.emitError()
                   << "measurements require simulate(..., rng)";
          }
          const auto q = walk.qubits->lookup(measureOp.getQubitIn());
          if (!q) {
            return measureOp.emitError()
                   << "qubit SSA value is not mapped for QCO DD construction";
          }
          if (deferred) {
            walk.classical->deferredMeasurements[measureOp.getResult()] = *q;
            walk.deferredMeasuredWires->insert(*q);
            walk.qubits->bind(measureOp.getQubitOut(), *q);
            return success();
          }
          const char bit = walk.dd->measureOneCollapsing(state, *q, *walk.rng);
          walk.classical->values[measureOp.getResult()] =
              BoolAttr::get(measureOp.getContext(), bit == '1');
          walk.qubits->bind(measureOp.getQubitOut(), *q);
          return success();
        }
      })
      .template Case<ResetOp>([&](ResetOp resetOp) -> LogicalResult {
        if constexpr (!std::is_same_v<StateDD, dd::VectorDD>) {
          return resetOp.emitError()
                 << "resets are not supported for QCO DD functionality "
                    "construction";
        } else {
          if (walk.rng == nullptr) {
            return resetOp.emitError() << "resets require simulate(..., rng)";
          }
          const auto q = walk.qubits->lookup(resetOp.getQubitIn());
          if (!q) {
            return resetOp.emitError()
                   << "qubit SSA value is not mapped for QCO DD construction";
          }
          const char bit = walk.dd->measureOneCollapsing(state, *q, *walk.rng);
          if (bit == '1') {
            state = walk.dd->applyOperation(
                walk.dd->makeGateDD(
                    dd::opToSingleQubitGateMatrix(dd::GateType::X), *q),
                state);
          }
          walk.qubits->bind(resetOp.getQubitOut(), *q);
          return success();
        }
      })
      .template Case<IfOp>([&](IfOp ifOp) -> LogicalResult {
        auto condition = lookupBool(ifOp.getCondition(), *walk.classical, ifOp);
        if (failed(condition)) {
          return failure();
        }
        Block* block = *condition ? ifOp.thenBlock() : ifOp.elseBlock();
        return applyRegionBranch(ifOp.getQubits(), *block,
                                 ifOp.getClassicalResults(),
                                 ifOp.getLinearResults(), walk, state, ifOp);
      })
      .template Case<IndexSwitchOp>(
          [&](IndexSwitchOp switchOp) -> LogicalResult {
            auto selector =
                lookupIndex(switchOp.getArg(), *walk.classical, switchOp);
            if (failed(selector)) {
              return failure();
            }
            Block* block = switchOp.getDefaultBlock();
            for (auto [i, caseValue] : llvm::enumerate(switchOp.getCases())) {
              if (caseValue == *selector) {
                block = switchOp.getCaseBlock(i);
                break;
              }
            }
            return applyRegionBranch(
                switchOp.getTargets(), *block, switchOp.getClassicalResults(),
                switchOp.getLinearResults(), walk, state, switchOp);
          })
      .template Case<scf::IfOp>([&](scf::IfOp ifOp) -> LogicalResult {
        auto condition = lookupBool(ifOp.getCondition(), *walk.classical, ifOp);
        if (failed(condition)) {
          return failure();
        }
        Region& selected =
            *condition ? ifOp.getThenRegion() : ifOp.getElseRegion();
        if (selected.empty()) {
          return ifOp.getNumResults() == 0
                     ? success()
                     : ifOp.emitError()
                           << "selected empty scf.if region has results";
        }
        return applyScfRegion(selected, ifOp.getResults(), walk, state, ifOp);
      })
      .template Case<scf::IndexSwitchOp>(
          [&](scf::IndexSwitchOp switchOp) -> LogicalResult {
            auto selector =
                lookupIndex(switchOp.getArg(), *walk.classical, switchOp);
            if (failed(selector)) {
              return failure();
            }
            Region* selected = &switchOp.getDefaultRegion();
            for (auto [i, value] : llvm::enumerate(switchOp.getCases())) {
              if (value == *selector) {
                selected = &switchOp.getCaseRegions()[i];
                break;
              }
            }
            return applyScfRegion(*selected, switchOp.getResults(), walk, state,
                                  switchOp);
          })
      .template Case<scf::ForOp>([&](scf::ForOp forOp) -> LogicalResult {
        auto range = resolveLoop(forOp, *walk.classical);
        if (failed(range)) {
          return failure();
        }

        Block& body = *forOp.getBody();
        SmallVector<Value> carried(forOp.getInits().begin(),
                                   forOp.getInits().end());

        for (size_t t = 0; t < range->trips;
             ++t, range->induction += range->step) {
          if (failed(consumeExecutionStep(walk, forOp))) {
            return failure();
          }
          auto iterArgs = body.getArguments().drop_front();
          if (failed(bindValuePairs(carried, iterArgs, walk, forOp))) {
            return failure();
          }
          if (failed(bindInteger(
                  body.getArgument(0),
                  range->induction.trunc(range->induction.getBitWidth() - 1),
                  *walk.classical))) {
            return failure();
          }
          if (failed(walkBlock(body, walk, state))) {
            return failure();
          }
          auto yield = cast<scf::YieldOp>(body.getTerminator());
          carried.assign(yield.getOperands().begin(),
                         yield.getOperands().end());
        }
        return bindValuePairs(carried, forOp.getResults(), walk, forOp);
      })
      .template Case<scf::WhileOp>([&](scf::WhileOp whileOp) -> LogicalResult {
        Block& before = whileOp.getBefore().front();
        Block& after = whileOp.getAfter().front();
        SmallVector<Value> carried(whileOp.getInits().begin(),
                                   whileOp.getInits().end());
        while (true) {
          if (failed(bindValuePairs(carried, before.getArguments(), walk,
                                    whileOp)) ||
              failed(walkBlock(before, walk, state))) {
            return failure();
          }
          auto condition = whileOp.getConditionOp();
          auto value =
              lookupBool(condition.getCondition(), *walk.classical, whileOp);
          if (failed(value)) {
            return failure();
          }
          if (!*value) {
            return bindValuePairs(condition.getArgs(), whileOp.getResults(),
                                  walk, whileOp);
          }
          if (failed(consumeExecutionStep(walk, whileOp))) {
            return failure();
          }
          if (failed(bindValuePairs(condition.getArgs(), after.getArguments(),
                                    walk, whileOp)) ||
              failed(walkBlock(after, walk, state))) {
            return failure();
          }
          auto yield = whileOp.getYieldOp();
          carried.assign(yield.getOperands().begin(),
                         yield.getOperands().end());
        }
      })
      .template Case<func::CallOp>([&](func::CallOp call) -> LogicalResult {
        auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            call, call.getCalleeAttr());
        if (!callee) {
          return call.emitError() << "func.call callee '" << call.getCallee()
                                  << "' could not be resolved";
        }
        if (callee.isDeclaration()) {
          return call.emitError() << "func.call callee must have a body";
        }
        Operation* calleeOp = callee.getOperation();
        if (!walk.activeCalls.insert(calleeOp).second) {
          return call.emitError()
                 << "recursive func.call is not supported for QCO DD "
                    "simulation";
        }
        const auto guard =
            llvm::make_scope_exit([&] { walk.activeCalls.erase(calleeOp); });

        if (failed(consumeExecutionStep(walk, call))) {
          return failure();
        }

        if (failed(bindValuePairs(call.getArgOperands(), callee.getArguments(),
                                  walk, call))) {
          return failure();
        }

        auto returnOp = walkFunctionBody(callee, walk, state);
        if (failed(returnOp)) {
          return failure();
        }

        /// Map callee return operands onto call results via the return op.
        return bindValuePairs(returnOp->getOperands(), call.getResults(), walk,
                              call);
      })
      .template Case<CtrlOp>([&](CtrlOp ctrlOp) -> LogicalResult {
        if (failed(checkDeferredMeasurementUse(ctrlOp, walk))) {
          return failure();
        }
        if (auto inner = mqt::getSoleBodyUnitary<UnitaryOpInterface>(
                *ctrlOp.getBody())) {
          auto decoded = decodeStandardGate(inner, *walk.classical);
          if (failed(decoded)) {
            return failure();
          }
          if (*decoded) {
            auto controlQubits =
                walk.qubits->lookupRange(ctrlOp.getControlsIn(), ctrlOp);
            if (failed(controlQubits)) {
              return failure();
            }
            dd::Controls controls;
            for (dd::Qubit q : *controlQubits) {
              controls.emplace(q);
            }
            return applyDecodedStandard(ctrlOp, **decoded, controls, walk,
                                        state);
          }
        }
        return applyUnitaryMatrix(ctrlOp, walk, state);
      })
      .template Case<UnitaryOpInterface>(
          [&](UnitaryOpInterface unitary) -> LogicalResult {
            if (failed(checkDeferredMeasurementUse(unitary, walk))) {
              return failure();
            }
            auto decoded = decodeStandardGate(unitary, *walk.classical);
            if (failed(decoded)) {
              return failure();
            }
            if (*decoded) {
              return applyDecodedStandard(unitary, **decoded, {}, walk, state);
            }
            return applyUnitaryMatrix(unitary, walk, state);
          })
      .Default([&](Operation* unsupported) -> LogicalResult {
        const StringRef dialect = unsupported->getName().getDialectNamespace();
        if (dialect == arith::ArithDialect::getDialectNamespace() ||
            dialect == math::MathDialect::getDialectNamespace()) {
          return applyClassicalOp(*unsupported, *walk.classical);
        }
        return unsupported->emitError()
               << "unsupported op for QCO DD construction: "
               << unsupported->getName().getStringRef();
      });
}

template <typename StateDD>
static FailureOr<func::ReturnOp>
walkFunctionBody(func::FuncOp func, WalkState& walk, StateDD& state) {
  if (!func.getBody().hasOneBlock()) {
    return func.emitError() << "QCO DD execution requires one-block functions";
  }
  Block& block = func.getBody().front();
  if (failed(walkBlock(block, walk, state))) {
    return failure();
  }
  return cast<func::ReturnOp>(block.getTerminator());
}

template <typename StateDD>
static LogicalResult walkFunction(func::FuncOp func, WalkState& walk,
                                  StateDD& state) {
  auto returnOp = walkFunctionBody(func, walk, state);
  if (failed(returnOp)) {
    return failure();
  }
  return validateReturn(*returnOp, *walk.qubits, *walk.tensors);
}

namespace {
struct PreparedState {
  QubitMap qubits;
  TensorMap tensors;
  ClassicalEnv classical;
};
} // namespace

static FailureOr<PreparedState>
prepare(func::FuncOp func, dd::Package& dd,
        const DDArgumentBindings& argumentBindings,
        bool bindEntryAllocations = false) {
  if (func.isDeclaration() || !func.getBody().hasOneBlock()) {
    return func.emitError()
           << "QCO DD execution requires a one-block function body";
  }

  PreparedState prepared;
  if (failed(
          applyArgumentBindings(func, argumentBindings, prepared.classical))) {
    return failure();
  }
  QubitMap& qubits = prepared.qubits;
  for (StaticOp staticOp : func.getBody().front().getOps<StaticOp>()) {
    const auto index = static_cast<size_t>(staticOp.getIndex());
    if (index >= dd::Package::MAX_POSSIBLE_QUBITS) {
      return staticOp.emitError()
             << "static qubit index exceeds the supported qubit range";
    }
    const auto q = static_cast<dd::Qubit>(index);
    qubits.bind(staticOp.getQubit(), q);
    qubits.numQubits = std::max(qubits.numQubits, static_cast<size_t>(q) + 1);
  }
  if (qubits.numQubits == 0) {
    size_t next = 0;
    for (Value arg : func.getArguments()) {
      if (isa<QubitType>(arg.getType())) {
        if (next >= dd::Package::MAX_POSSIBLE_QUBITS) {
          return func.emitError()
                 << "QCO function exceeds the supported qubit range";
        }
        qubits.bind(arg, static_cast<dd::Qubit>(next++));
      } else if (isQTensorType(arg.getType())) {
        const auto type = cast<RankedTensorType>(arg.getType());
        int64_t size = type.getDimSize(0);
        if (type.isDynamicDim(0)) {
          const auto binding = argumentBindings.find(arg);
          if (binding == argumentBindings.end()) {
            return func.emitError()
                   << "dynamic qtensor arguments require an index extent";
          }
          size = cast<IntegerAttr>(binding->second).getValue().getSExtValue();
          if (size < 0) {
            return func.emitError()
                   << "dynamic qtensor extent must be non-negative";
          }
        }
        const auto count = static_cast<size_t>(size);
        if (count > dd::Package::MAX_POSSIBLE_QUBITS - next) {
          return func.emitError()
                 << "QCO function exceeds the supported qubit range";
        }
        TensorSlots slots;
        slots.reserve(count);
        for (size_t i = 0; i < count; ++i) {
          slots.emplace_back(static_cast<dd::Qubit>(next++));
        }
        prepared.tensors.bind(arg,
                              std::make_shared<TensorSlots>(std::move(slots)));
      }
    }
    qubits.numQubits = next;
  }
  if (bindEntryAllocations) {
    for (AllocOp alloc : func.getBody().front().getOps<AllocOp>()) {
      if (qubits.numQubits >= dd::Package::MAX_POSSIBLE_QUBITS) {
        return alloc.emitError()
               << "QCO function exceeds the supported qubit range";
      }
      qubits.bind(alloc.getResult(),
                  static_cast<dd::Qubit>(qubits.numQubits++));
    }
  }
  if (dd.qubits() < qubits.numQubits) {
    dd.resize(qubits.numQubits);
  }
  return prepared;
}

FailureOr<dd::MatrixDD>
buildFunctionality(func::FuncOp func, dd::Package& dd,
                   const DDArgumentBindings& argumentBindings) {
  auto prepared =
      prepare(func, dd, argumentBindings, /*bindEntryAllocations=*/true);
  if (failed(prepared)) {
    return failure();
  }
  QubitMap qubits = std::move(prepared->qubits);
  TensorMap tensors = std::move(prepared->tensors);
  ClassicalEnv classical = std::move(prepared->classical);
  WalkState walkState{.qubits = &qubits,
                      .tensors = &tensors,
                      .classical = &classical,
                      .dd = &dd,
                      .rng = nullptr};
  walkState.activeCalls.insert(func.getOperation());

  dd::MatrixDD state =
      qubits.numQubits == 0
          ? dd::MatrixDD::one()
          : dd.createInitialMatrix(std::vector<bool>(qubits.numQubits, false));
  if (failed(walkFunction(func, walkState, state))) {
    if (qubits.numQubits != 0) {
      dd.decRef(state);
    }
    return failure();
  }
  return state;
}

static FailureOr<dd::VectorDD>
simulateImpl(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd,
             const PreparedState& prepared, std::mt19937_64* rng,
             const DenseSet<Operation*>* deferredMeasurements = nullptr,
             ClassicalEnv* finalClassical = nullptr,
             DenseSet<dd::Qubit>* deferredMeasuredWires = nullptr,
             Operation** deferredMeasurementUse = nullptr,
             bool validateQuantumReturn = true) {
  const size_t inputQubits =
      in.isTerminal() ? 0U : static_cast<size_t>(in.p->v) + 1U;
  if (inputQubits < prepared.qubits.numQubits) {
    dd.decRef(in);
    return func.emitError()
           << "input state has " << inputQubits << " qubits but function uses "
           << prepared.qubits.numQubits;
  }
  QubitMap qubits = prepared.qubits;
  qubits.numQubits = inputQubits;
  TensorMap tensors = prepared.tensors.clone();
  ClassicalEnv classical = prepared.classical;
  classical.deferredMeasurementUse = deferredMeasurementUse;
  WalkState walkState{.qubits = &qubits,
                      .tensors = &tensors,
                      .classical = &classical,
                      .dd = &dd,
                      .rng = rng,
                      .deferredMeasurements = deferredMeasurements,
                      .deferredMeasuredWires = deferredMeasuredWires};
  walkState.activeCalls.insert(func.getOperation());

  dd::VectorDD state = in;
  auto returnOp = walkFunctionBody(func, walkState, state);
  if (failed(returnOp) ||
      (validateQuantumReturn &&
       failed(validateReturn(*returnOp, qubits, tensors)))) {
    dd.decRef(state);
    return failure();
  }
  if (finalClassical != nullptr) {
    *finalClassical = std::move(classical);
  }
  return state;
}

FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd, std::mt19937_64& rng,
                                 const DDArgumentBindings& argumentBindings) {
  auto prepared = prepare(func, dd, argumentBindings);
  if (failed(prepared)) {
    dd.decRef(in);
    return failure();
  }
  return simulateImpl(func, in, dd, *prepared, &rng);
}

static bool mayMeasureOrReset(func::FuncOp func, DenseSet<Operation*>& active) {
  if (!active.insert(func).second) {
    return true;
  }
  const auto guard = llvm::make_scope_exit([&] { active.erase(func); });
  bool found = false;
  func.getBody().walk([&](Operation* op) {
    if (found || isa<MeasureOp, ResetOp>(op)) {
      found = true;
      return;
    }
    auto call = dyn_cast<func::CallOp>(op);
    if (!call) {
      return;
    }
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    found =
        !callee || callee.isDeclaration() || mayMeasureOrReset(callee, active);
  });
  return found;
}

static void analyzeSampling(func::FuncOp func, SamplingPlan& plan) {
  func.getBody().walk([&](Operation* op) {
    if (isa<ResetOp>(op)) {
      plan.dynamic = true;
      return;
    }
    if (isa<MeasureOp>(op)) {
      plan.deferredMeasurements.insert(op);
      return;
    }
    auto call = dyn_cast<func::CallOp>(op);
    if (!call) {
      return;
    }
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    DenseSet<Operation*> active;
    if (!callee || callee.isDeclaration() ||
        mayMeasureOrReset(callee, active)) {
      plan.dynamic = true;
    }
  });
}

static FailureOr<SamplingPlan>
getSamplingPlan(func::FuncOp func, bool statevectorAnalysis = false) {
  Block& entry = func.getBody().front();
  SamplingPlan plan;
  auto returnOp = dyn_cast<func::ReturnOp>(entry.getTerminator());
  if (!returnOp) {
    return func.emitError()
           << "single-block QCO DD execution requires func.return";
  }

  bool hasOther = false;
  for (Value value : returnOp.getOperands()) {
    if (isa<cbit::RegisterType>(value.getType())) {
      plan.outputs.push_back(value);
    } else {
      hasOther = true;
    }
  }
  if (!statevectorAnalysis && !plan.outputs.empty() && hasOther) {
    return returnOp.emitError()
           << "QCO DD sampling does not support mixed CBit and non-CBit "
              "results";
  }

  analyzeSampling(func, plan);
  return plan;
}

FailureOr<dd::VectorDD>
simulateStatevector(func::FuncOp func, dd::Package& dd,
                    const DDArgumentBindings& argumentBindings) {
  auto prepared = prepare(func, dd, argumentBindings);
  if (failed(prepared)) {
    return failure();
  }
  auto plan = getSamplingPlan(func, /*statevectorAnalysis=*/true);
  if (failed(plan)) {
    return failure();
  }
  if (plan->dynamic) {
    return func.emitError()
           << "statevector extraction supports only terminal measurements "
              "that assemble returned CBit registers";
  }
  DenseSet<dd::Qubit> measuredWires;
  Operation* deferredMeasurementUse = nullptr;
  auto state = simulateImpl(
      func, dd::makeZeroState(prepared->qubits.numQubits, dd), dd, *prepared,
      nullptr, &plan->deferredMeasurements, nullptr, &measuredWires,
      &deferredMeasurementUse, /*validateQuantumReturn=*/false);
  if (failed(state) && deferredMeasurementUse != nullptr) {
    return deferredMeasurementUse->emitError()
           << "statevector extraction cannot use a measurement result or "
              "measured qubit before program end";
  }
  return state;
}

static FailureOr<std::string> encodeOutcome(ArrayRef<Value> outputs,
                                            const ClassicalEnv& classical,
                                            StringRef basis) {
  if (outputs.empty()) {
    return basis.str();
  }
  std::string outcome;
  for (Value value : outputs) {
    const auto reg = classical.registers.find(value);
    if (reg == classical.registers.end()) {
      return emitError(value.getLoc())
             << "returned CBit register is not mapped for QCO DD simulation";
    }
    for (size_t i = reg->second->size(); i > 0; --i) {
      const size_t index = i - 1;
      const auto& cell = (*reg->second)[index];
      if (cell.value) {
        outcome.push_back(*cell.value ? '1' : '0');
      } else if (cell.deferredWire && *cell.deferredWire < basis.size()) {
        outcome.push_back(basis[basis.size() - 1 - *cell.deferredWire]);
      } else {
        return emitError(value.getLoc())
               << "returned CBit register element " << index << " is undefined";
      }
    }
  }
  return outcome;
}

static FailureOr<std::map<std::string, size_t>>
sampleImpl(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd,
           size_t shots, std::mt19937_64& rng, const PreparedState& prepared) {
  const auto inputGuard = llvm::make_scope_exit([&] { dd.decRef(in); });
  auto plan = getSamplingPlan(func);
  if (failed(plan)) {
    return failure();
  }

  const size_t inputQubits =
      in.isTerminal() ? 0U : static_cast<size_t>(in.p->v) + 1U;
  if (inputQubits < prepared.qubits.numQubits) {
    return func.emitError()
           << "input state has " << inputQubits << " qubits but function uses "
           << prepared.qubits.numQubits;
  }

  std::map<std::string, size_t> counts;
  if (shots == 0) {
    return counts;
  }

  const auto record = [&](const ClassicalEnv& classical,
                          StringRef basis) -> LogicalResult {
    auto outcome = encodeOutcome(plan->outputs, classical, basis);
    if (failed(outcome)) {
      return failure();
    }
    ++counts[*outcome];
    return success();
  };

  if (!plan->dynamic) {
    ClassicalEnv classical;
    DenseSet<dd::Qubit> measuredWires;
    Operation* deferredMeasurementUse = nullptr;
    dd.incRef(in);
    auto state = simulateImpl(func, in, dd, prepared, nullptr,
                              &plan->deferredMeasurements, &classical,
                              &measuredWires, &deferredMeasurementUse);
    if (succeeded(state)) {
      const auto guard = llvm::make_scope_exit([&] { dd.decRef(*state); });
      for (size_t i = 0; i < shots; ++i) {
        if (failed(record(classical, dd.measureAll(*state, false, rng)))) {
          return failure();
        }
      }
      return counts;
    }
    if (deferredMeasurementUse == nullptr) {
      return failure();
    }
  }

  for (size_t i = 0; i < shots; ++i) {
    ClassicalEnv classical;
    dd.incRef(in);
    auto state =
        simulateImpl(func, in, dd, prepared, &rng, nullptr, &classical);
    if (failed(state)) {
      return failure();
    }
    const auto guard = llvm::make_scope_exit([&] { dd.decRef(*state); });
    const std::string basis = plan->outputs.empty()
                                  ? dd.measureAll(*state, false, rng)
                                  : std::string{};
    if (failed(record(classical, basis))) {
      return failure();
    }
  }
  return counts;
}

FailureOr<std::map<std::string, size_t>>
sample(func::FuncOp func, size_t shots, uint64_t seed,
       const DDArgumentBindings& argumentBindings) {
  dd::Package dd;
  std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
  auto prepared = prepare(func, dd, argumentBindings);
  if (failed(prepared)) {
    return failure();
  }
  return sampleImpl(func, dd::makeZeroState(prepared->qubits.numQubits, dd), dd,
                    shots, rng, *prepared);
}

} // namespace mlir::qco
