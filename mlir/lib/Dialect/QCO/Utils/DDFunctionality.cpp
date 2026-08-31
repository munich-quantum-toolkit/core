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
#include "ir/Definitions.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
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
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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
  DenseMap<Value, qc::Qubit> qubits;
  size_t numQubits = 0;

  void bind(Value value, qc::Qubit q) { qubits[value] = q; }

  [[nodiscard]] std::optional<qc::Qubit> lookup(Value value) const {
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

  FailureOr<SmallVector<qc::Qubit>> lookupRange(ValueRange values,
                                                Operation* op) const {
    SmallVector<qc::Qubit> out;
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
using TensorSlots = SmallVector<std::optional<qc::Qubit>>;

struct TensorMap {
  DenseMap<Value, TensorSlots> tensors;

  void bind(Value value, TensorSlots slots) {
    tensors[value] = std::move(slots);
  }

  [[nodiscard]] const TensorSlots* lookup(Value value) const {
    const auto it = tensors.find(value);
    return it == tensors.end() ? nullptr : &it->second;
  }

  void erase(Value value) { tensors.erase(value); }
};

struct ClassicalEnv {
  struct RegisterBit {
    std::optional<bool> value;
    std::optional<qc::Qubit> deferredWire;
  };
  using RegisterState = std::vector<RegisterBit>;
  using Scalar = std::variant<bool, int64_t, llvm::APInt, double>;

  DenseMap<Value, Scalar> values;
  DenseMap<Value, qc::Qubit> deferredMeasurements;
  /// Shared storage preserves CBit register identity across `func.call`.
  DenseMap<Value, std::shared_ptr<RegisterState>> registers;
  /// Shared storage preserves caller-visible writes through `func.call`.
  DenseMap<Value, std::shared_ptr<SmallVector<Scalar>>> memrefs;

  LogicalResult bindFrom(Value source, Value dest, Operation* op) {
    const auto it = values.find(source);
    if (it == values.end()) {
      return op->emitError()
             << "classical SSA value is not mapped for QCO DD simulation";
    }
    values[dest] = it->second;
    return success();
  }
};

struct DecodedGate {
  qc::OpType type = qc::OpType::None;
  std::vector<dd::fp> params;
};

struct WalkState {
  QubitMap* qubits;
  TensorMap* tensors;
  ClassicalEnv* classical;
  dd::Package* dd;
  std::mt19937_64* rng = nullptr;
  const DenseSet<Operation*>* deferredMeasurements = nullptr;
  size_t remainingExecutionSteps = MAX_CONTROL_FLOW_STEPS;
  DenseSet<Operation*> activeCalls;
};
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

[[nodiscard]] static bool isQTensorType(Type type) {
  const auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getRank() == 1 &&
         isa<QubitType>(tensorType.getElementType());
}

template <typename T>
static FailureOr<T> lookupScalar(Value value, const ClassicalEnv& classical,
                                 Operation* op) {
  const auto it = classical.values.find(value);
  if (it == classical.values.end() || !std::holds_alternative<T>(it->second)) {
    return op->emitError()
           << "classical SSA value is not mapped for QCO DD simulation";
  }
  return std::get<T>(it->second);
}

static FailureOr<double>
resolveDouble(Value value, const ClassicalEnv& classical, Operation* op) {
  if (const auto it = classical.values.find(value);
      it != classical.values.end() &&
      std::holds_alternative<double>(it->second)) {
    return std::get<double>(it->second);
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
  const auto type =
      TypeSwitch<Operation*, qc::OpType>(op)
          .Case<IdOp>([](auto) { return qc::OpType::I; })
          .Case<XOp>([](auto) { return qc::OpType::X; })
          .Case<YOp>([](auto) { return qc::OpType::Y; })
          .Case<ZOp>([](auto) { return qc::OpType::Z; })
          .Case<HOp>([](auto) { return qc::OpType::H; })
          .Case<SOp>([](auto) { return qc::OpType::S; })
          .Case<SdgOp>([](auto) { return qc::OpType::Sdg; })
          .Case<TOp>([](auto) { return qc::OpType::T; })
          .Case<TdgOp>([](auto) { return qc::OpType::Tdg; })
          .Case<SXOp>([](auto) { return qc::OpType::SX; })
          .Case<SXdgOp>([](auto) { return qc::OpType::SXdg; })
          .Case<RXOp>([](auto) { return qc::OpType::RX; })
          .Case<RYOp>([](auto) { return qc::OpType::RY; })
          .Case<RZOp>([](auto) { return qc::OpType::RZ; })
          .Case<POp>([](auto) { return qc::OpType::P; })
          .Case<ROp>([](auto) { return qc::OpType::R; })
          .Case<U2Op>([](auto) { return qc::OpType::U2; })
          .Case<UOp>([](auto) { return qc::OpType::U; })
          .Case<SWAPOp>([](auto) { return qc::OpType::SWAP; })
          .Case<iSWAPOp>([](auto) { return qc::OpType::iSWAP; })
          .Case<DCXOp>([](auto) { return qc::OpType::DCX; })
          .Case<ECROp>([](auto) { return qc::OpType::ECR; })
          .Case<RCCXOp>([](auto) { return qc::OpType::RCCX; })
          .Case<RXXOp>([](auto) { return qc::OpType::RXX; })
          .Case<RYYOp>([](auto) { return qc::OpType::RYY; })
          .Case<RZZOp>([](auto) { return qc::OpType::RZZ; })
          .Case<RZXOp>([](auto) { return qc::OpType::RZX; })
          .Case<XXPlusYYOp>([](auto) { return qc::OpType::XXplusYY; })
          .Case<XXMinusYYOp>([](auto) { return qc::OpType::XXminusYY; })
          .Default([](auto) { return qc::OpType::None; });
  if (type == qc::OpType::None) {
    return std::optional<DecodedGate>{std::nullopt};
  }
  DecodedGate decoded{.type = type, .params = {}};
  for (Value param : unitary.getParameters()) {
    auto concrete = resolveDouble(param, classical, op);
    if (failed(concrete)) {
      return failure();
    }
    decoded.params.push_back(static_cast<dd::fp>(*concrete));
  }
  return std::optional{std::move(decoded)};
}

static dd::mCachedEdge
buildEmbeddedLocalDD(dd::Package& dd, const DynamicMatrix& local,
                     const DenseMap<qc::Qubit, size_t>& operandForWire,
                     size_t numOperands, int64_t level, size_t row,
                     size_t col) {
  if (level < 0) {
    return dd::mCachedEdge::terminal(
        local(static_cast<int64_t>(row), static_cast<int64_t>(col)));
  }
  const auto wire = static_cast<qc::Qubit>(level);
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
                                        ArrayRef<qc::Qubit> wires) {
  DenseMap<qc::Qubit, size_t> operandForWire;
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
  ArrayRef<qc::Qubit> wires = *wiresOr;
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
                                          const qc::Controls& controls,
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
      getStandardOperationDD(*walk.dd, gate.type, gate.params, controls,
                             {targets->begin(), targets->end()}),
      state);
  return walk.qubits->remapUnitary(unitary);
}

static LogicalResult validateReturn(func::ReturnOp returnOp,
                                    const QubitMap& qubits,
                                    const TensorMap& tensors) {
  qc::Qubit expected = 0;
  for (Value value : returnOp.getOperands()) {
    if (isQTensorType(value.getType())) {
      const auto* slots = tensors.lookup(value);
      if (slots == nullptr) {
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

static LogicalResult recordConstant(arith::ConstantOp constant,
                                    ClassicalEnv& classical) {
  if (auto attr = dyn_cast<BoolAttr>(constant.getValue())) {
    classical.values[constant.getResult()] = attr.getValue();
  } else if (auto attr = dyn_cast<FloatAttr>(constant.getValue())) {
    if (!constant.getType().isF64()) {
      return constant.emitError()
             << "QCO DD simulation only supports f64 classical values";
    }
    classical.values[constant.getResult()] = attr.getValue().convertToDouble();
  } else if (auto attr = dyn_cast<IntegerAttr>(constant.getValue())) {
    if (constant.getType().isInteger(1)) {
      classical.values[constant.getResult()] = attr.getValue() != 0;
    } else if (isa<IndexType>(constant.getType())) {
      classical.values[constant.getResult()] = attr.getInt();
    } else if (isa<IntegerType>(constant.getType())) {
      classical.values[constant.getResult()] = attr.getValue();
    }
  }
  return success();
}

static LogicalResult applyBindings(func::FuncOp func,
                                   const DDBindings& bindings,
                                   ClassicalEnv& classical) {
  for (const auto& [value, attr] : bindings) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &func.getBody().front()) {
      return func.emitError()
             << "QCO DD bindings must target entry-block arguments";
    }
    const Type type = value.getType();
    if (isQTensorType(type)) {
      if (cast<RankedTensorType>(type).isDynamicDim(0) &&
          isa<IntegerAttr>(attr)) {
        continue;
      }
    } else if (type.isInteger(1)) {
      if (auto boolean = dyn_cast<BoolAttr>(attr)) {
        classical.values[value] = boolean.getValue();
        continue;
      }
      if (auto integer = dyn_cast<IntegerAttr>(attr)) {
        classical.values[value] = integer.getValue() != 0;
        continue;
      }
    } else if (isa<IndexType>(type)) {
      if (auto integer = dyn_cast<IntegerAttr>(attr)) {
        classical.values[value] = integer.getInt();
        continue;
      }
    } else if (auto integerType = dyn_cast<IntegerType>(type)) {
      if (auto integer = dyn_cast<IntegerAttr>(attr)) {
        classical.values[value] =
            integer.getValue().sextOrTrunc(integerType.getWidth());
        continue;
      }
    } else if (type.isF64()) {
      if (auto floating = dyn_cast<FloatAttr>(attr);
          floating && floating.getType().isF64()) {
        classical.values[value] = floating.getValue().convertToDouble();
        continue;
      }
    }
    return func.emitError() << "QCO DD binding attribute " << attr
                            << " does not match argument type " << type;
  }
  return success();
}

static FailureOr<bool> lookupBool(Value value, const ClassicalEnv& classical,
                                  Operation* op) {
  return lookupScalar<bool>(value, classical, op);
}

static FailureOr<int64_t>
lookupIndex(Value value, const ClassicalEnv& classical, Operation* op) {
  return lookupScalar<int64_t>(value, classical, op);
}

static FailureOr<double> lookupFloat(Value value, const ClassicalEnv& classical,
                                     Operation* op) {
  return lookupScalar<double>(value, classical, op);
}

static FailureOr<llvm::APInt>
lookupInteger(Value value, const ClassicalEnv& classical, Operation* op) {
  if (value.getType().isInteger(1)) {
    auto bit = lookupBool(value, classical, op);
    if (failed(bit)) {
      return failure();
    }
    return llvm::APInt(1, static_cast<uint64_t>(*bit));
  }
  if (isa<IndexType>(value.getType())) {
    auto index = lookupIndex(value, classical, op);
    if (failed(index)) {
      return failure();
    }
    return llvm::APInt(64, static_cast<uint64_t>(*index));
  }
  if (isa<IntegerType>(value.getType())) {
    return lookupScalar<llvm::APInt>(value, classical, op);
  }
  return op->emitError() << "expected an integer or index SSA value";
}

[[nodiscard]] static bool evaluateCmp(arith::CmpIPredicate predicate,
                                      const llvm::APInt& lhs,
                                      const llvm::APInt& rhs) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return lhs == rhs;
  case arith::CmpIPredicate::ne:
    return lhs != rhs;
  case arith::CmpIPredicate::slt:
    return lhs.slt(rhs);
  case arith::CmpIPredicate::sle:
    return lhs.sle(rhs);
  case arith::CmpIPredicate::sgt:
    return lhs.sgt(rhs);
  case arith::CmpIPredicate::sge:
    return lhs.sge(rhs);
  case arith::CmpIPredicate::ult:
    return lhs.ult(rhs);
  case arith::CmpIPredicate::ule:
    return lhs.ule(rhs);
  case arith::CmpIPredicate::ugt:
    return lhs.ugt(rhs);
  case arith::CmpIPredicate::uge:
    return lhs.uge(rhs);
  }
  llvm_unreachable("unknown arith.cmpi predicate");
}

static LogicalResult bindInteger(Value dest, const llvm::APInt& value,
                                 ClassicalEnv& classical) {
  if (dest.getType().isInteger(1)) {
    classical.values[dest] = value[0];
  } else if (isa<IndexType>(dest.getType())) {
    classical.values[dest] = static_cast<int64_t>(value.getZExtValue());
  } else if (auto type = dyn_cast<IntegerType>(dest.getType())) {
    classical.values[dest] = value.zextOrTrunc(type.getWidth());
  } else {
    return failure();
  }
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
  if (!cell.value) {
    return load.emitError() << "read from an undefined CBit register element";
  }
  return bindInteger(load.getResult(),
                     llvm::APInt(1, static_cast<uint64_t>(*cell.value)),
                     classical);
}

[[nodiscard]] static bool isSupportedClassicalType(Type type) {
  return isa<IndexType, IntegerType>(type) || type.isF64();
}

static FailureOr<ClassicalEnv::Scalar*>
lookupMemRefSlot(Value memref, ValueRange indices, ClassicalEnv& classical,
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

static ClassicalEnv::Scalar zeroScalar(Type type) {
  if (type.isInteger(1)) {
    return false;
  }
  if (isa<IndexType>(type)) {
    return int64_t{0};
  }
  if (auto integer = dyn_cast<IntegerType>(type)) {
    return llvm::APInt(integer.getWidth(), 0);
  }
  return 0.0;
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
    if (alloc.getDynamicSizes().size() != 1) {
      return alloc.emitError() << "dynamic 1-D memref requires one size";
    }
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
      std::make_shared<SmallVector<ClassicalEnv::Scalar>>(
          static_cast<size_t>(size), zeroScalar(type.getElementType()));
  return success();
}

static LogicalResult applyMemRefStore(memref::StoreOp store,
                                      ClassicalEnv& classical) {
  auto slot =
      lookupMemRefSlot(store.getMemref(), store.getIndices(), classical, store);
  const auto value = classical.values.find(store.getValue());
  if (failed(slot) || value == classical.values.end()) {
    if (value == classical.values.end()) {
      store.emitError()
          << "stored classical value is not mapped for QCO DD simulation";
    }
    return failure();
  }
  **slot = value->second;
  return success();
}

static LogicalResult applyMemRefLoad(memref::LoadOp load,
                                     ClassicalEnv& classical) {
  auto slot =
      lookupMemRefSlot(load.getMemref(), load.getIndices(), classical, load);
  if (failed(slot)) {
    return failure();
  }
  classical.values[load.getResult()] = **slot;
  return success();
}

template <typename OpTy, typename Combine>
static LogicalResult applyBinaryInteger(OpTy op, ClassicalEnv& classical,
                                        Combine combine) {
  auto lhs = lookupInteger(op.getLhs(), classical, op);
  auto rhs = lookupInteger(op.getRhs(), classical, op);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  return bindInteger(op.getResult(), combine(*lhs, *rhs), classical);
}

template <typename OpTy, typename Combine>
static LogicalResult applyBinaryFloat(OpTy op, ClassicalEnv& classical,
                                      Combine combine) {
  auto lhs = lookupFloat(op.getLhs(), classical, op);
  auto rhs = lookupFloat(op.getRhs(), classical, op);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  classical.values[op.getResult()] = combine(*lhs, *rhs);
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

template <typename OpTy, typename Shift>
static LogicalResult applyShift(OpTy op, ClassicalEnv& classical, Shift shift) {
  auto lhs = lookupInteger(op.getLhs(), classical, op);
  auto rhs = lookupInteger(op.getRhs(), classical, op);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  if (rhs->uge(lhs->getBitWidth())) {
    return op.emitError() << "shift amount out of range for QCO DD simulation";
  }
  return bindInteger(op.getResult(), shift(*lhs, rhs->getZExtValue()),
                     classical);
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
      .Case<arith::AndIOp>([&](auto value) {
        return applyBinaryInteger(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs & rhs;
            });
      })
      .Case<arith::OrIOp>([&](auto value) {
        return applyBinaryInteger(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs | rhs;
            });
      })
      .Case<arith::XOrIOp>([&](auto value) {
        return applyBinaryInteger(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs ^ rhs;
            });
      })
      .Case<arith::AddIOp>([&](auto value) {
        return applyBinaryInteger(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs + rhs;
            });
      })
      .Case<arith::SubIOp>([&](auto value) {
        return applyBinaryInteger(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs - rhs;
            });
      })
      .Case<arith::MulIOp>([&](auto value) {
        return applyBinaryInteger(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs * rhs;
            });
      })
      .Case<arith::DivUIOp>([&](auto value) {
        return applyDivision(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs.udiv(rhs);
            });
      })
      .Case<arith::DivSIOp>([&](auto value) {
        return applyDivision(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs.sdiv(rhs);
            });
      })
      .Case<arith::RemUIOp>([&](auto value) {
        return applyDivision(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs.urem(rhs);
            });
      })
      .Case<arith::RemSIOp>([&](auto value) {
        return applyDivision(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs.srem(rhs);
            });
      })
      .Case<arith::ShLIOp>([&](auto value) {
        return applyShift(
            value, classical,
            [](const llvm::APInt& lhs, uint64_t rhs) { return lhs.shl(rhs); });
      })
      .Case<arith::ShRUIOp>([&](auto value) {
        return applyShift(
            value, classical,
            [](const llvm::APInt& lhs, uint64_t rhs) { return lhs.lshr(rhs); });
      })
      .Case<arith::ShRSIOp>([&](auto value) {
        return applyShift(
            value, classical,
            [](const llvm::APInt& lhs, uint64_t rhs) { return lhs.ashr(rhs); });
      })
      .Case<arith::CmpIOp>([&](arith::CmpIOp cmp) -> LogicalResult {
        auto lhs = lookupInteger(cmp.getLhs(), classical, cmp);
        auto rhs = lookupInteger(cmp.getRhs(), classical, cmp);
        if (failed(lhs) || failed(rhs)) {
          return failure();
        }
        classical.values[cmp.getResult()] =
            evaluateCmp(cmp.getPredicate(), *lhs, *rhs);
        return success();
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
      .Case<arith::AddFOp>([&](auto value) {
        return applyBinaryFloat(
            value, classical, [](double lhs, double rhs) { return lhs + rhs; });
      })
      .Case<arith::SubFOp>([&](auto value) {
        return applyBinaryFloat(
            value, classical, [](double lhs, double rhs) { return lhs - rhs; });
      })
      .Case<arith::MulFOp>([&](auto value) {
        return applyBinaryFloat(
            value, classical, [](double lhs, double rhs) { return lhs * rhs; });
      })
      .Case<arith::DivFOp>([&](auto value) {
        return applyBinaryFloat(
            value, classical, [](double lhs, double rhs) { return lhs / rhs; });
      })
      .Case<arith::RemFOp>([&](auto value) {
        return applyBinaryFloat(value, classical, [](double lhs, double rhs) {
          return std::fmod(lhs, rhs);
        });
      })
      .Case<arith::NegFOp>([&](arith::NegFOp neg) -> LogicalResult {
        auto value = lookupFloat(neg.getOperand(), classical, neg);
        if (failed(value)) {
          return failure();
        }
        classical.values[neg.getResult()] = -*value;
        return success();
      })
      .Case<arith::CmpFOp>([&](arith::CmpFOp cmp) -> LogicalResult {
        auto lhs = lookupFloat(cmp.getLhs(), classical, cmp);
        auto rhs = lookupFloat(cmp.getRhs(), classical, cmp);
        if (failed(lhs) || failed(rhs)) {
          return failure();
        }
        classical.values[cmp.getResult()] = arith::applyCmpPredicate(
            cmp.getPredicate(), llvm::APFloat(*lhs), llvm::APFloat(*rhs));
        return success();
      })
      .Case<arith::SIToFPOp, arith::UIToFPOp>(
          [&](Operation* castOp) -> LogicalResult {
            auto value =
                lookupInteger(castOp->getOperand(0), classical, castOp);
            if (failed(value)) {
              return failure();
            }
            classical.values[castOp->getResult(0)] =
                value->roundToDouble(isa<arith::SIToFPOp>(castOp));
            return success();
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

static FailureOr<LoopRange>
resolveLoop(scf::ForOp forOp, ClassicalEnv& classical, size_t remainingSteps) {
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
  const size_t limited = trips.getLimitedValue(remainingSteps + 1);
  if (limited > remainingSteps) {
    return forOp.emitError(
        "QCO DD execution exceeds the limit of 10000 control-flow steps");
  }
  return LoopRange{.induction = lowerWide, .step = stepWide, .trips = limited};
}

static LogicalResult bindValuePairs(ValueRange sources, ValueRange dests,
                                    WalkState& walk, Operation* op);

static LogicalResult bindLinearArgs(ValueRange operands, Block& block,
                                    WalkState& walk, Operation* op) {
  for (Value arg : block.getArguments()) {
    if (!isa<QubitType>(arg.getType()) && !isQTensorType(arg.getType())) {
      return op->emitError()
             << "unsupported linear region argument for QCO DD simulation";
    }
  }
  return bindValuePairs(operands, block.getArguments(), walk, op);
}

static LogicalResult bindValuePairs(ValueRange sources, ValueRange dests,
                                    WalkState& walk, Operation* op) {
  const QubitMap sourceQubits = *walk.qubits;
  const TensorMap sourceTensors = *walk.tensors;
  const ClassicalEnv sourceClassical = *walk.classical;
  for (auto [src, dest] : llvm::zip_equal(sources, dests)) {
    if (isa<QubitType>(dest.getType())) {
      const auto q = sourceQubits.lookup(src);
      if (!q) {
        return op->emitError()
               << "qubit SSA value is not mapped for QCO DD construction";
      }
      walk.qubits->bind(dest, *q);
    } else if (isQTensorType(dest.getType())) {
      const auto* slots = sourceTensors.lookup(src);
      if (slots == nullptr) {
        return op->emitError()
               << "qtensor SSA value is not mapped for QCO DD simulation";
      }
      walk.tensors->bind(dest, *slots);
    } else if (isa<cbit::RegisterType>(dest.getType())) {
      const auto it = sourceClassical.registers.find(src);
      if (it == sourceClassical.registers.end()) {
        return op->emitError()
               << "CBit register is not mapped for QCO DD simulation";
      }
      walk.classical->registers[dest] = it->second;
    } else if (isa<MemRefType>(dest.getType())) {
      const auto it = sourceClassical.memrefs.find(src);
      if (it == sourceClassical.memrefs.end()) {
        return op->emitError()
               << "classical memref is not mapped for QCO DD simulation";
      }
      walk.classical->memrefs[dest] = it->second;
    } else {
      const auto value = sourceClassical.values.find(src);
      if (value == sourceClassical.values.end()) {
        return op->emitError()
               << "classical SSA value is not mapped for QCO DD simulation";
      }
      walk.classical->values[dest] = value->second;
    }
  }
  return success();
}

static LogicalResult bindYieldResults(YieldOp yield,
                                      ValueRange classicalResults,
                                      ValueRange linearResults,
                                      WalkState& walk) {
  const size_t numClassical = classicalResults.size();
  if (yield.getNumOperands() != numClassical + linearResults.size()) {
    return yield.emitError()
           << "yield operand count does not match operation results";
  }
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
  if (failed(bindLinearArgs(linearOperands, block, walk, parent))) {
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
  if (!region.hasOneBlock()) {
    return parent->emitError()
           << "SCF region must contain exactly one block for QCO DD simulation";
  }
  Block& block = region.front();
  if (failed(walkBlock(block, walk, state))) {
    return failure();
  }
  auto yield = dyn_cast<scf::YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != results.size()) {
    return parent->emitError()
           << "SCF region must yield one value for each result";
  }
  return bindValuePairs(yield.getOperands(), results, walk, parent);
}

static FailureOr<TensorSlots> allocateZeroQubits(size_t count, WalkState& walk,
                                                 dd::VectorDD& state,
                                                 Operation* op) {
  if (count == 0) {
    return op->emitError() << "quantum allocation size must be positive";
  }
  if (walk.qubits->numQubits > walk.dd->qubits() ||
      count > walk.dd->qubits() - walk.qubits->numQubits) {
    return op->emitError() << "DD package has " << walk.dd->qubits()
                           << " qubits but allocation requires "
                           << walk.qubits->numQubits + count;
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
    slots.emplace_back(static_cast<qc::Qubit>(first + i));
  }
  walk.qubits->numQubits += count;
  return slots;
}

template <typename StateDD>
static LogicalResult applyOp(Operation& op, WalkState& walk, StateDD& state) {
  return TypeSwitch<Operation*, LogicalResult>(&op)
      .template Case<StaticOp, SinkOp>([](auto) { return success(); })
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
              walk.tensors->bind(alloc.getResult(), std::move(*slots));
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
            for (const qc::Qubit wire : *wires) {
              slots.emplace_back(wire);
            }
            walk.tensors->bind(fromElements.getResult(), std::move(slots));
            return success();
          })
      .template Case<qtensor::ExtractOp>(
          [&](qtensor::ExtractOp extract) -> LogicalResult {
            const auto* input = walk.tensors->lookup(extract.getTensor());
            auto index =
                lookupIndex(extract.getIndex(), *walk.classical, extract);
            if (input == nullptr || failed(index)) {
              if (input == nullptr) {
                extract.emitError()
                    << "qtensor is not mapped for QCO DD simulation";
              }
              return failure();
            }
            if (*index < 0 || static_cast<size_t>(*index) >= input->size()) {
              return extract.emitError() << "qtensor index out of range";
            }
            TensorSlots output = *input;
            auto& wire = output[static_cast<size_t>(*index)];
            if (!wire) {
              return extract.emitError()
                     << "qtensor element has already been extracted";
            }
            walk.qubits->bind(extract.getResult(), *wire);
            wire.reset();
            walk.tensors->bind(extract.getOutTensor(), std::move(output));
            return success();
          })
      .template Case<qtensor::InsertOp>(
          [&](qtensor::InsertOp insert) -> LogicalResult {
            const auto* input = walk.tensors->lookup(insert.getDest());
            const auto wire = walk.qubits->lookup(insert.getScalar());
            auto index =
                lookupIndex(insert.getIndex(), *walk.classical, insert);
            if (input == nullptr || !wire || failed(index)) {
              if (input == nullptr || !wire) {
                insert.emitError()
                    << "qtensor or qubit is not mapped for QCO DD simulation";
              }
              return failure();
            }
            if (*index < 0 || static_cast<size_t>(*index) >= input->size()) {
              return insert.emitError() << "qtensor index out of range";
            }
            TensorSlots output = *input;
            output[static_cast<size_t>(*index)] = wire;
            walk.tensors->bind(insert.getResult(), std::move(output));
            return success();
          })
      .template Case<qtensor::DeallocOp>(
          [&](qtensor::DeallocOp dealloc) -> LogicalResult {
            if (walk.tensors->lookup(dealloc.getTensor()) == nullptr) {
              return dealloc.emitError()
                     << "qtensor is not mapped for QCO DD simulation";
            }
            walk.tensors->erase(dealloc.getTensor());
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
      .template Case<
          arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::AddIOp,
          arith::SubIOp, arith::MulIOp, arith::DivUIOp, arith::DivSIOp,
          arith::RemUIOp, arith::RemSIOp, arith::ShLIOp, arith::ShRUIOp,
          arith::ShRSIOp, arith::CmpIOp, arith::SelectOp, arith::ExtUIOp,
          arith::ExtSIOp, arith::IndexCastUIOp, arith::IndexCastOp,
          arith::TruncIOp, arith::AddFOp, arith::SubFOp, arith::MulFOp,
          arith::DivFOp, arith::RemFOp, arith::NegFOp, arith::CmpFOp,
          arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp>(
          [&](Operation* classicalOp) {
            return applyClassicalOp(*classicalOp, *walk.classical);
          })
      .template Case<func::ReturnOp>([&](func::ReturnOp returnOp) {
        return validateReturn(returnOp, *walk.qubits, *walk.tensors);
      })
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
            walk.qubits->bind(measureOp.getQubitOut(), *q);
            return success();
          }
          const char bit = walk.dd->measureOneCollapsing(state, *q, *walk.rng);
          walk.classical->values[measureOp.getResult()] = bit == '1';
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
                    dd::opToSingleQubitGateMatrix(qc::OpType::X), *q),
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
        if (block == nullptr) {
          return ifOp.emitError() << "selected qco.if region is empty";
        }
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
            if (block == nullptr) {
              return switchOp.emitError()
                     << "selected qco.index_switch region is empty";
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
      .template Case<scf::ExecuteRegionOp>(
          [&](scf::ExecuteRegionOp execute) -> LogicalResult {
            return applyScfRegion(execute.getRegion(), execute.getResults(),
                                  walk, state, execute);
          })
      .template Case<scf::ForOp>([&](scf::ForOp forOp) -> LogicalResult {
        auto range =
            resolveLoop(forOp, *walk.classical, walk.remainingExecutionSteps);
        if (failed(range)) {
          return failure();
        }

        Block& body = *forOp.getBody();
        SmallVector<Value> carried(forOp.getInits().begin(),
                                   forOp.getInits().end());

        for (size_t t = 0; t < range->trips;
             ++t, range->induction += range->step) {
          if (walk.remainingExecutionSteps == 0) {
            return forOp.emitError(
                "QCO DD execution exceeds the limit of 10000 control-flow "
                "steps");
          }
          --walk.remainingExecutionSteps;
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
        if (!whileOp.getBefore().hasOneBlock() ||
            !whileOp.getAfter().hasOneBlock()) {
          return whileOp.emitError()
                 << "scf.while regions must contain one block";
        }
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
          auto condition = dyn_cast<scf::ConditionOp>(before.getTerminator());
          if (!condition) {
            return whileOp.emitError()
                   << "scf.while before region missing scf.condition";
          }
          auto value =
              lookupBool(condition.getCondition(), *walk.classical, whileOp);
          if (failed(value)) {
            return failure();
          }
          if (!*value) {
            return bindValuePairs(condition.getArgs(), whileOp.getResults(),
                                  walk, whileOp);
          }
          if (walk.remainingExecutionSteps == 0) {
            return whileOp.emitError(
                "QCO DD execution exceeds the limit of 10000 control-flow "
                "steps");
          }
          --walk.remainingExecutionSteps;
          if (failed(bindValuePairs(condition.getArgs(), after.getArguments(),
                                    walk, whileOp)) ||
              failed(walkBlock(after, walk, state))) {
            return failure();
          }
          auto yield = dyn_cast<scf::YieldOp>(after.getTerminator());
          if (!yield) {
            return whileOp.emitError()
                   << "scf.while after region missing scf.yield";
          }
          carried.assign(yield.getOperands().begin(),
                         yield.getOperands().end());
        }
      })
      .template Case<func::CallOp>([&](func::CallOp call) -> LogicalResult {
        auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            call, call.getCalleeAttr());
        if (!callee.getBody().hasOneBlock()) {
          return call.emitError()
                 << "func.call callee must have a single-block body";
        }
        auto returnOp =
            cast<func::ReturnOp>(callee.getBody().front().getTerminator());
        Operation* calleeOp = callee.getOperation();
        if (!walk.activeCalls.insert(calleeOp).second) {
          return call.emitError()
                 << "recursive func.call is not supported for QCO DD "
                    "simulation";
        }
        const auto guard =
            llvm::make_scope_exit([&] { walk.activeCalls.erase(calleeOp); });

        if (failed(bindValuePairs(call.getArgOperands(), callee.getArguments(),
                                  walk, call))) {
          return failure();
        }

        if (failed(walkBlock(callee.getBody().front(), walk, state))) {
          return failure();
        }
        return bindValuePairs(returnOp.getOperands(), call.getResults(), walk,
                              call);
      })
      .template Case<CtrlOp>([&](CtrlOp ctrlOp) -> LogicalResult {
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
            qc::Controls controls;
            for (qc::Qubit q : *controlQubits) {
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
        if (unsupported->getName().getDialectNamespace() ==
            arith::ArithDialect::getDialectNamespace()) {
          return applyClassicalOp(*unsupported, *walk.classical);
        }
        return unsupported->emitError()
               << "unsupported op for QCO DD construction: "
               << unsupported->getName().getStringRef();
      });
}

template <typename StateDD>
static LogicalResult walkFunction(func::FuncOp func, WalkState& walkState,
                                  StateDD& state) {
  walkState.activeCalls.insert(func.getOperation());
  // Function bodies include `func.return` as terminator; region walks skip
  // `qco.yield` and bind it separately.
  for (Operation& op : func.getBody().front()) {
    if (failed(applyOp(op, walkState, state))) {
      return failure();
    }
  }
  return success();
}

namespace {
struct PreparedState {
  QubitMap qubits;
  TensorMap tensors;
};
} // namespace

static FailureOr<PreparedState> prepare(func::FuncOp func,
                                        const dd::Package& dd,
                                        const DDBindings& bindings,
                                        bool bindEntryAllocations = false) {
  if (!func.getBody().hasOneBlock()) {
    return func.emitError()
           << "QCO DD construction expects a single-block function body";
  }

  PreparedState prepared;
  QubitMap& qubits = prepared.qubits;
  for (StaticOp staticOp : func.getBody().front().getOps<StaticOp>()) {
    const auto q = static_cast<qc::Qubit>(staticOp.getIndex());
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
        qubits.bind(arg, static_cast<qc::Qubit>(next++));
      } else if (isQTensorType(arg.getType())) {
        const auto type = cast<RankedTensorType>(arg.getType());
        int64_t size = type.getDimSize(0);
        if (type.isDynamicDim(0)) {
          const auto binding = bindings.find(arg);
          if (binding == bindings.end() || !isa<IntegerAttr>(binding->second)) {
            return func.emitError()
                   << "dynamic qtensor arguments require an integer extent";
          }
          size = cast<IntegerAttr>(binding->second).getInt();
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
          slots.emplace_back(static_cast<qc::Qubit>(next++));
        }
        prepared.tensors.bind(arg, std::move(slots));
      }
    }
    qubits.numQubits = next;
  }
  if (bindEntryAllocations) {
    for (AllocOp alloc : func.getBody().front().getOps<AllocOp>()) {
      qubits.bind(alloc.getResult(),
                  static_cast<qc::Qubit>(qubits.numQubits++));
    }
  }
  if (dd.qubits() < qubits.numQubits) {
    return func.emitError() << "DD package has " << dd.qubits()
                            << " qubits but function uses " << qubits.numQubits;
  }
  return prepared;
}

FailureOr<dd::MatrixDD> buildFunctionality(func::FuncOp func, dd::Package& dd,
                                           const DDBindings& bindings) {
  auto prepared = prepare(func, dd, bindings, /*bindEntryAllocations=*/true);
  if (failed(prepared)) {
    return failure();
  }
  QubitMap qubits = std::move(prepared->qubits);
  TensorMap tensors = std::move(prepared->tensors);
  ClassicalEnv classical;
  if (failed(applyBindings(func, bindings, classical))) {
    return failure();
  }
  WalkState walkState{.qubits = &qubits,
                      .tensors = &tensors,
                      .classical = &classical,
                      .dd = &dd,
                      .rng = nullptr};

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
             const DDBindings& bindings,
             const DenseSet<Operation*>* deferredMeasurements = nullptr,
             ClassicalEnv* finalClassical = nullptr) {
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
  TensorMap tensors = prepared.tensors;
  ClassicalEnv classical;
  if (failed(applyBindings(func, bindings, classical))) {
    dd.decRef(in);
    return failure();
  }
  WalkState walkState{.qubits = &qubits,
                      .tensors = &tensors,
                      .classical = &classical,
                      .dd = &dd,
                      .rng = rng,
                      .deferredMeasurements = deferredMeasurements};

  dd::VectorDD state = in;
  if (failed(walkFunction(func, walkState, state))) {
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
                                 const DDBindings& bindings) {
  auto prepared = prepare(func, dd, bindings);
  if (failed(prepared)) {
    dd.decRef(in);
    return failure();
  }
  return simulateImpl(func, in, dd, *prepared, &rng, bindings);
}

static bool isOutputOnlyRegister(Value reg, ArrayRef<Value> outputs) {
  return llvm::is_contained(outputs, reg) &&
         llvm::all_of(reg.getUses(), [reg](const OpOperand& use) {
           if (auto store = dyn_cast<cbit::StoreOp>(use.getOwner())) {
             return store.getReg() == reg;
           }
           return isa<func::ReturnOp>(use.getOwner());
         });
}

static bool hasOutputOnlyMeasurementResult(MeasureOp measure,
                                           ArrayRef<Value> outputs) {
  if (measure.getResult().use_empty()) {
    return false;
  }
  return llvm::all_of(measure.getResult().getUses(), [&](const OpOperand& use) {
    auto store = dyn_cast<cbit::StoreOp>(use.getOwner());
    return store && isOutputOnlyRegister(store.getReg(), outputs);
  });
}

static std::optional<size_t> getConstantTensorIndex(Value value) {
  const auto attr = mqt::valueToConstantAttr(value);
  const auto integer = attr ? dyn_cast<IntegerAttr>(*attr) : IntegerAttr{};
  if (!integer || integer.getInt() < 0) {
    return std::nullopt;
  }
  return static_cast<size_t>(integer.getInt());
}

static bool hasOnlyTerminalQuantumUses(Value value,
                                       std::optional<size_t> tensorSlot,
                                       func::FuncOp func,
                                       ArrayRef<Value> outputs,
                                       DenseSet<Value>& visited) {
  if (!visited.insert(value).second) {
    return true;
  }
  return llvm::all_of(value.getUses(), [&](OpOperand& use) {
    Operation* owner = use.getOwner();
    if (isa<SinkOp, func::ReturnOp, qtensor::DeallocOp>(owner)) {
      return true;
    }
    if (auto measure = dyn_cast<MeasureOp>(owner)) {
      return !tensorSlot && measure->getParentOfType<func::FuncOp>() == func &&
             hasOutputOnlyMeasurementResult(measure, outputs) &&
             hasOnlyTerminalQuantumUses(measure.getQubitOut(), std::nullopt,
                                        func, outputs, visited);
    }
    if (auto fromElements = dyn_cast<qtensor::FromElementsOp>(owner)) {
      return !tensorSlot && hasOnlyTerminalQuantumUses(fromElements.getResult(),
                                                       use.getOperandNumber(),
                                                       func, outputs, visited);
    }
    if (auto insert = dyn_cast<qtensor::InsertOp>(owner)) {
      const auto index = getConstantTensorIndex(insert.getIndex());
      if (!index) {
        return false;
      }
      if (use.get() == insert.getScalar()) {
        return !tensorSlot &&
               hasOnlyTerminalQuantumUses(insert.getResult(), *index, func,
                                          outputs, visited);
      }
      return tensorSlot && *tensorSlot != *index &&
             hasOnlyTerminalQuantumUses(insert.getResult(), tensorSlot, func,
                                        outputs, visited);
    }
    if (auto extract = dyn_cast<qtensor::ExtractOp>(owner)) {
      const auto index = getConstantTensorIndex(extract.getIndex());
      if (!tensorSlot || !index) {
        return false;
      }
      if (*tensorSlot == *index) {
        return hasOnlyTerminalQuantumUses(extract.getResult(), std::nullopt,
                                          func, outputs, visited);
      }
      return hasOnlyTerminalQuantumUses(extract.getOutTensor(), tensorSlot,
                                        func, outputs, visited);
    }
    return false;
  });
}

static bool isDeferrableMeasurement(MeasureOp measure, func::FuncOp func,
                                    ArrayRef<Value> outputs) {
  if (!hasOutputOnlyMeasurementResult(measure, outputs)) {
    return false;
  }
  DenseSet<Value> visited;
  return hasOnlyTerminalQuantumUses(measure.getQubitOut(), std::nullopt, func,
                                    outputs, visited);
}

static void analyzeSampling(func::FuncOp func, ArrayRef<Value> outputs,
                            DenseSet<Operation*>& active, SamplingPlan& plan) {
  Operation* funcOp = func.getOperation();
  if (!active.insert(funcOp).second) {
    plan.dynamic = true;
    return;
  }
  func.getBody().walk([&](Operation* op) {
    if (isa<ResetOp>(op)) {
      plan.dynamic = true;
    } else if (auto measure = dyn_cast<MeasureOp>(op)) {
      if (isDeferrableMeasurement(measure, func, outputs)) {
        plan.deferredMeasurements.insert(op);
      } else {
        plan.dynamic = true;
      }
    } else if (auto call = dyn_cast<func::CallOp>(op)) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || callee.isDeclaration() ||
          !callee.getBody().hasOneBlock()) {
        plan.dynamic = true;
      } else {
        analyzeSampling(callee, outputs, active, plan);
      }
    }
  });
  active.erase(funcOp);
}

static FailureOr<SamplingPlan> getSamplingPlan(func::FuncOp func) {
  Block& entry = func.getBody().front();
  auto returnOp = cast<func::ReturnOp>(entry.getTerminator());
  SamplingPlan plan;
  bool hasOther = false;
  for (Value value : returnOp.getOperands()) {
    if (isa<cbit::RegisterType>(value.getType())) {
      plan.outputs.push_back(value);
    } else {
      hasOther = true;
    }
  }
  if (!plan.outputs.empty() && hasOther) {
    return returnOp.emitError()
           << "QCO DD sampling does not support mixed CBit and non-CBit "
              "results";
  }

  DenseSet<Operation*> active;
  analyzeSampling(func, plan.outputs, active, plan);
  return plan;
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

FailureOr<std::map<std::string, size_t>> sample(func::FuncOp func,
                                                dd::Package& dd, size_t shots,
                                                std::mt19937_64& rng,
                                                const DDBindings& bindings) {
  auto prepared = prepare(func, dd, bindings);
  if (failed(prepared)) {
    return failure();
  }
  auto plan = getSamplingPlan(func);
  if (failed(plan)) {
    return failure();
  }

  std::map<std::string, size_t> counts;
  if (shots == 0) {
    return counts;
  }

  const size_t numQubits = prepared->qubits.numQubits;
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
    auto state = simulateImpl(func, dd::makeZeroState(numQubits, dd), dd,
                              *prepared, nullptr, bindings,
                              &plan->deferredMeasurements, &classical);
    if (failed(state)) {
      return failure();
    }
    const auto guard = llvm::make_scope_exit([&] { dd.decRef(*state); });
    for (size_t i = 0; i < shots; ++i) {
      if (failed(record(classical, dd.measureAll(*state, false, rng)))) {
        return failure();
      }
    }
    return counts;
  }

  for (size_t i = 0; i < shots; ++i) {
    ClassicalEnv classical;
    auto state = simulateImpl(func, dd::makeZeroState(numQubits, dd), dd,
                              *prepared, &rng, bindings, nullptr, &classical);
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

} // namespace mlir::qco
