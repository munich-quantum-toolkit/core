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
#include "dd/Complex.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
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
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
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
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
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

  void releaseWire(const qc::Qubit released) {
    SmallVector<Value> aliases;
    for (auto& [value, wire] : qubits) {
      if (wire == released) {
        aliases.push_back(value);
      } else if (wire > released) {
        --wire;
      }
    }
    for (Value alias : aliases) {
      qubits.erase(alias);
    }
    --numQubits;
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

  void releaseWire(const qc::Qubit released) {
    for (auto& [value, slots] : tensors) {
      (void)value;
      for (auto& wire : slots) {
        if (!wire) {
          continue;
        }
        if (*wire == released) {
          wire.reset();
        } else if (*wire > released) {
          --*wire;
        }
      }
    }
  }
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

  void releaseWire(const qc::Qubit released) {
    for (auto& [value, wire] : deferredMeasurements) {
      (void)value;
      if (wire > released) {
        --wire;
      }
    }

    DenseSet<RegisterState*> updated;
    for (auto& [value, reg] : registers) {
      (void)value;
      if (!updated.insert(reg.get()).second) {
        continue;
      }
      for (auto& cell : *reg) {
        if (cell.deferredWire && *cell.deferredWire > released) {
          --*cell.deferredWire;
        }
      }
    }
  }

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

enum class DeallocationMode : std::uint8_t {
  Apply,
  PreserveAll,
  PreserveDeferred,
};

struct WalkState {
  QubitMap* qubits;
  TensorMap* tensors;
  ClassicalEnv* classical;
  dd::Package* dd;
  std::mt19937_64* rng = nullptr;
  const DenseSet<Operation*>* deferredMeasurements = nullptr;
  DeallocationMode deallocationMode = DeallocationMode::Apply;
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

/// Distinguishes a density operator from a functionality matrix.
struct DensityState {
  dd::MatrixDD matrix;
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

static dd::mCachedEdge
buildDensityMatrix(const dd::VectorDD& ket, const dd::VectorDD& bra,
                   const int64_t level, dd::Package& dd,
                   std::map<std::tuple<dd::vNode*, dd::vNode*, int64_t>,
                            dd::mCachedEdge>& cache) {
  if (ket.isZeroTerminal() || bra.isZeroTerminal()) {
    return dd::mCachedEdge::zero();
  }
  const auto weight =
      static_cast<dd::ComplexValue>(ket.w) * dd::ComplexNumbers::conj(bra.w);
  if (level < 0) {
    return dd::mCachedEdge::terminal(weight);
  }

  const auto key = std::tuple{ket.p, bra.p, level};
  if (const auto cached = cache.find(key); cached != cache.end()) {
    return {cached->second.p, cached->second.w * weight};
  }
  const auto child = [level](const dd::VectorDD& edge,
                             const size_t index) -> dd::VectorDD {
    if (!edge.isTerminal() && std::cmp_equal(edge.p->v, level)) {
      return edge.p->e[index];
    }
    return index == 0 ? dd::VectorDD{.p = edge.p, .w = dd::Complex::one()}
                      : dd::VectorDD::zero();
  };
  const auto ketZero = child(ket, 0);
  const auto ketOne = child(ket, 1);
  const auto braZero = child(bra, 0);
  const auto braOne = child(bra, 1);
  auto result = dd.makeDDNode<dd::mNode, dd::CachedEdge>(
      static_cast<qc::Qubit>(level),
      {buildDensityMatrix(ketZero, braZero, level - 1, dd, cache),
       buildDensityMatrix(ketZero, braOne, level - 1, dd, cache),
       buildDensityMatrix(ketOne, braZero, level - 1, dd, cache),
       buildDensityMatrix(ketOne, braOne, level - 1, dd, cache)});
  cache.try_emplace(key, result);
  result.w = result.w * weight;
  return result;
}

static void applyStateOperation(const dd::MatrixDD& operation, dd::Package& dd,
                                dd::VectorDD& state) {
  state = dd.applyOperation(operation, state);
}

static void applyStateOperation(const dd::MatrixDD& operation, dd::Package& dd,
                                dd::MatrixDD& state) {
  state = dd.applyOperation(operation, state);
}

static void applyStateOperation(const dd::MatrixDD& operation, dd::Package& dd,
                                DensityState& state) {
  const auto left = dd.multiply(operation, state.matrix);
  const auto adjoint = dd.conjugateTranspose(operation);
  auto result = dd.multiply(left, adjoint);
  dd.incRef(result);
  dd.decRef(state.matrix);
  state.matrix = result;
  dd.garbageCollect();
}

template <typename StateDD>
static LogicalResult applyUnitaryMatrix(UnitaryOpInterface unitary,
                                        WalkState& walk, StateDD& state) {
  Operation* op = unitary.getOperation();
  if (auto gphase = dyn_cast<GPhaseOp>(op)) {
    if constexpr (std::is_same_v<StateDD, DensityState>) {
      return success();
    }
    auto theta = resolveDouble(gphase.getTheta(), *walk.classical, op);
    if (failed(theta)) {
      return failure();
    }
    auto id = dd::Package::makeIdent();
    id.w = walk.dd->cn.lookup(std::cos(*theta), std::sin(*theta));
    applyStateOperation(id, *walk.dd, state);
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
    applyStateOperation(walk.dd->makeGateDD(mat, wires[0]), *walk.dd, state);
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
    applyStateOperation(walk.dd->makeTwoQubitGateDD(mat, wires[0], wires[1]),
                        *walk.dd, state);
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
    applyStateOperation(
        walk.dd->makeThreeQubitGateDD(mat, wires[0], wires[1], wires[2]),
        *walk.dd, state);
    return walk.qubits->remapUnitary(unitary);
  }

  applyStateOperation(
      makeEmbeddedLocalDD(*walk.dd, local, walk.qubits->numQubits, wires),
      *walk.dd, state);
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
  applyStateOperation(
      getStandardOperationDD(*walk.dd, gate.type, gate.params, controls,
                             {targets->begin(), targets->end()}),
      *walk.dd, state);
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

template <typename OpTy, typename Apply>
static LogicalResult applyUnaryFloat(OpTy op, ClassicalEnv& classical,
                                     Apply apply) {
  auto value = lookupFloat(op.getOperand(), classical, op);
  if (failed(value)) {
    return failure();
  }
  classical.values[op.getResult()] = apply(*value);
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
      .Case<arith::MaxSIOp>([&](auto value) {
        return applyBinaryInteger(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs.sgt(rhs) ? lhs : rhs;
            });
      })
      .Case<arith::MinSIOp>([&](auto value) {
        return applyBinaryInteger(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs.slt(rhs) ? lhs : rhs;
            });
      })
      .Case<arith::MaxUIOp>([&](auto value) {
        return applyBinaryInteger(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs.ugt(rhs) ? lhs : rhs;
            });
      })
      .Case<arith::MinUIOp>([&](auto value) {
        return applyBinaryInteger(
            value, classical,
            [](const llvm::APInt& lhs, const llvm::APInt& rhs) {
              return lhs.ult(rhs) ? lhs : rhs;
            });
      })
      .Case<arith::ShLIOp>([&](auto value) {
        return applyShift(value, classical,
                          [](const llvm::APInt& lhs, const uint64_t rhs) {
                            return lhs.shl(rhs);
                          });
      })
      .Case<arith::ShRUIOp>([&](auto value) {
        return applyShift(value, classical,
                          [](const llvm::APInt& lhs, const uint64_t rhs) {
                            return lhs.lshr(rhs);
                          });
      })
      .Case<arith::ShRSIOp>([&](auto value) {
        return applyShift(value, classical,
                          [](const llvm::APInt& lhs, const uint64_t rhs) {
                            return lhs.ashr(rhs);
                          });
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
      .Case<arith::MaximumFOp>([&](auto value) {
        return applyBinaryFloat(value, classical, [](double lhs, double rhs) {
          if (std::isnan(lhs) || std::isnan(rhs)) {
            return std::numeric_limits<double>::quiet_NaN();
          }
          return std::fmax(lhs, rhs);
        });
      })
      .Case<arith::MinimumFOp>([&](auto value) {
        return applyBinaryFloat(value, classical, [](double lhs, double rhs) {
          if (std::isnan(lhs) || std::isnan(rhs)) {
            return std::numeric_limits<double>::quiet_NaN();
          }
          return std::fmin(lhs, rhs);
        });
      })
      .Case<arith::MaxNumFOp>([&](auto value) {
        return applyBinaryFloat(value, classical, [](double lhs, double rhs) {
          return std::fmax(lhs, rhs);
        });
      })
      .Case<arith::MinNumFOp>([&](auto value) {
        return applyBinaryFloat(value, classical, [](double lhs, double rhs) {
          return std::fmin(lhs, rhs);
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
      .Case<math::AbsFOp>([&](auto value) {
        return applyUnaryFloat(
            value, classical, [](double operand) { return std::abs(operand); });
      })
      .Case<math::CeilOp>([&](auto value) {
        return applyUnaryFloat(value, classical, [](double operand) {
          return std::ceil(operand);
        });
      })
      .Case<math::CosOp>([&](auto value) {
        return applyUnaryFloat(
            value, classical, [](double operand) { return std::cos(operand); });
      })
      .Case<math::ExpOp>([&](auto value) {
        return applyUnaryFloat(
            value, classical, [](double operand) { return std::exp(operand); });
      })
      .Case<math::FloorOp>([&](auto value) {
        return applyUnaryFloat(value, classical, [](double operand) {
          return std::floor(operand);
        });
      })
      .Case<math::LogOp>([&](auto value) {
        return applyUnaryFloat(
            value, classical, [](double operand) { return std::log(operand); });
      })
      .Case<math::SinOp>([&](auto value) {
        return applyUnaryFloat(
            value, classical, [](double operand) { return std::sin(operand); });
      })
      .Case<math::SqrtOp>([&](auto value) {
        return applyUnaryFloat(value, classical, [](double operand) {
          return std::sqrt(operand);
        });
      })
      .Case<math::TanOp>([&](auto value) {
        return applyUnaryFloat(
            value, classical, [](double operand) { return std::tan(operand); });
      })
      .Case<math::PowFOp>([&](auto value) {
        return applyBinaryFloat(value, classical, [](double lhs, double rhs) {
          return std::pow(lhs, rhs);
        });
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
static FailureOr<Operation*>
walkConcreteCFG(Block& entry, WalkState& walk, StateDD& state,
                function_ref<bool(Operation*)> isExit, StringRef scope) {
  Block* block = &entry;
  while (true) {
    if (failed(walkBlock(*block, walk, state))) {
      return failure();
    }
    Operation* terminator = block->getTerminator();
    if (isExit(terminator)) {
      return terminator;
    }

    Block* successor = nullptr;
    ValueRange successorOperands;
    if (auto branch = dyn_cast<cf::BranchOp>(terminator)) {
      successor = branch.getDest();
      successorOperands = branch.getDestOperands();
    } else if (auto branch = dyn_cast<cf::CondBranchOp>(terminator)) {
      auto condition =
          lookupBool(branch.getCondition(), *walk.classical, branch);
      if (failed(condition)) {
        return failure();
      }
      successor = *condition ? branch.getTrueDest() : branch.getFalseDest();
      successorOperands = *condition ? branch.getTrueDestOperands()
                                     : branch.getFalseDestOperands();
    } else if (auto switchOp = dyn_cast<cf::SwitchOp>(terminator)) {
      auto flag = lookupInteger(switchOp.getFlag(), *walk.classical, switchOp);
      if (failed(flag)) {
        return failure();
      }
      successor = switchOp.getDefaultDestination();
      successorOperands = switchOp.getDefaultOperands();
      if (const auto caseValues = switchOp.getCaseValues()) {
        for (auto [caseValue, destination, operands] : llvm::zip_equal(
                 caseValues->getValues<llvm::APInt>(),
                 switchOp.getCaseDestinations(), switchOp.getCaseOperands())) {
          if (*flag == caseValue) {
            successor = destination;
            successorOperands = operands;
            break;
          }
        }
      }
    } else {
      return terminator->emitError() << "unsupported " << scope
                                     << " CFG terminator for QCO DD simulation";
    }
    if (successorOperands.size() != successor->getNumArguments()) {
      return terminator->emitError()
             << scope
             << " CFG successor operand count does not match block arguments";
    }
    if (failed(bindValuePairs(successorOperands, successor->getArguments(),
                              walk, terminator))) {
      return failure();
    }
    if (walk.remainingExecutionSteps == 0) {
      return terminator->emitError()
             << "QCO DD execution exceeds the limit of 10000 control-flow "
                "steps";
    }
    --walk.remainingExecutionSteps;
    block = successor;
  }
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
  if (region.empty()) {
    return parent->emitError() << "SCF region is empty";
  }
  auto terminator = walkConcreteCFG(
      region.front(), walk, state,
      [](Operation* op) { return isa<scf::YieldOp>(op); }, "SCF region");
  if (failed(terminator)) {
    return failure();
  }
  auto yield = dyn_cast<scf::YieldOp>(*terminator);
  if (!yield || yield.getNumOperands() != results.size()) {
    return parent->emitError()
           << "SCF region must yield one value for each result";
  }
  return bindValuePairs(yield.getOperands(), results, walk, parent);
}

template <typename StateDD>
static FailureOr<TensorSlots> allocateZeroQubits(size_t count, WalkState& walk,
                                                 StateDD& state,
                                                 Operation* op) {
  static_assert(std::is_same_v<StateDD, dd::VectorDD> ||
                std::is_same_v<StateDD, DensityState>);
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
  if constexpr (std::is_same_v<StateDD, dd::VectorDD>) {
    auto zeros = dd::makeZeroState(count, *walk.dd, first);
    auto extended = walk.dd->kronecker(zeros, state, first, /*incIdx=*/false);
    walk.dd->incRef(extended);
    walk.dd->decRef(zeros);
    walk.dd->decRef(state);
    state = extended;
  } else {
    auto extended = state.matrix;
    for (size_t i = 0; i < count; ++i) {
      extended = walk.dd->makeDDNode<dd::mNode, dd::Edge>(
          static_cast<qc::Qubit>(first + i),
          {extended, dd::MatrixDD::zero(), dd::MatrixDD::zero(),
           dd::MatrixDD::zero()});
    }
    walk.dd->incRef(extended);
    walk.dd->decRef(state.matrix);
    state.matrix = extended;
    walk.dd->garbageCollect();
  }

  TensorSlots slots;
  slots.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    slots.emplace_back(static_cast<qc::Qubit>(first + i));
  }
  walk.qubits->numQubits += count;
  return slots;
}

/// Project @p wire onto one basis state and remove its DD level.
static dd::VectorDD projectAndRemoveWire(const dd::VectorDD& root,
                                         const qc::Qubit wire,
                                         const bool projectOne,
                                         dd::Package& dd) {
  DenseMap<dd::vNode*, dd::VectorDD> projectedNodes;
  const auto project = [&](const auto& self,
                           const dd::VectorDD& edge) -> dd::VectorDD {
    if (edge.isZeroTerminal()) {
      return edge;
    }
    // A skipped vector-DD level represents a qubit fixed to zero.
    if (edge.isTerminal() || edge.p->v < wire) {
      return projectOne ? dd::VectorDD::zero() : edge;
    }

    dd::VectorDD projected;
    if (const auto cached = projectedNodes.find(edge.p);
        cached != projectedNodes.end()) {
      projected = cached->second;
    } else if (edge.p->v == wire) {
      projected = edge.p->e[projectOne ? 1U : 0U];
    } else {
      std::array<dd::VectorDD, dd::RADIX> edges{self(self, edge.p->e[0]),
                                                self(self, edge.p->e[1])};
      projected = dd.makeDDNode<dd::vNode, dd::Edge>(
          static_cast<qc::Qubit>(edge.p->v - 1U), edges);
    }
    projectedNodes.try_emplace(edge.p, projected);
    projected.w = dd.cn.lookup(projected.w * edge.w);
    return projected;
  };
  return project(project, root);
}

static LogicalResult deallocateWire(const qc::Qubit wire, WalkState& walk,
                                    dd::VectorDD& state, Operation* op) {
  if (wire >= walk.qubits->numQubits) {
    return op->emitError()
           << "deallocated wire is outside the simulated register";
  }
  const auto zero = projectAndRemoveWire(state, wire, false, *walk.dd);
  const auto one = projectAndRemoveWire(state, wire, true, *walk.dd);
  if (zero.isZeroTerminal() && one.isZeroTerminal()) {
    return op->emitError() << "cannot deallocate a zero-norm quantum state";
  }
  if (!zero.isZeroTerminal() && !one.isZeroTerminal() && zero.p != one.p) {
    return op->emitError()
           << "deallocating an entangled qubit is not supported by "
              "statevector QCO DD simulation";
  }

  const auto zeroWeight = static_cast<dd::ComplexValue>(zero.w);
  const auto oneWeight = static_cast<dd::ComplexValue>(one.w);
  const auto norm = std::sqrt(zeroWeight.mag2() + oneWeight.mag2());
  auto reduced = dd::VectorDD{.p = zero.isZeroTerminal() ? one.p : zero.p,
                              .w = walk.dd->cn.lookup(norm)};
  walk.dd->incRef(reduced);
  walk.dd->decRef(state);
  state = reduced;
  walk.qubits->releaseWire(wire);
  walk.tensors->releaseWire(wire);
  walk.classical->releaseWire(wire);
  return success();
}

static LogicalResult deallocateWire(const qc::Qubit wire, WalkState& walk,
                                    DensityState& state, Operation* op) {
  if (wire >= walk.qubits->numQubits) {
    return op->emitError()
           << "deallocated wire is outside the simulated register";
  }
  std::vector<bool> eliminate(walk.qubits->numQubits, false);
  eliminate[wire] = true;
  auto reduced = walk.dd->partialTrace(state.matrix, eliminate);
  // `partialTrace` normalizes by two for each removed matrix level. A physical
  // partial trace does not, so restore the removed factor.
  reduced.w =
      walk.dd->cn.lookup(static_cast<dd::ComplexValue>(reduced.w) * 2.0);
  walk.dd->incRef(reduced);
  walk.dd->decRef(state.matrix);
  state.matrix = reduced;
  walk.dd->garbageCollect();
  walk.qubits->releaseWire(wire);
  walk.tensors->releaseWire(wire);
  return success();
}

static double densityTrace(const dd::MatrixDD& density, const size_t numQubits,
                           dd::Package& dd) {
  const auto normalized = dd.trace(density, numQubits);
  return std::ldexp(normalized.r, static_cast<int>(numQubits));
}

static FailureOr<char> measureDensity(DensityState& state, const qc::Qubit wire,
                                      const size_t numQubits, dd::Package& dd,
                                      std::mt19937_64& rng,
                                      Operation* diagnosticOp) {
  const auto project = [&](const dd::GateMatrix& projector) {
    const auto gate = dd.makeGateDD(projector, wire);
    return dd.multiply(dd.multiply(gate, state.matrix), gate);
  };
  auto zero = project(dd::MEAS_ZERO_MAT);
  auto one = project(dd::MEAS_ONE_MAT);
  const double rawZero = densityTrace(zero, numQubits, dd);
  const double rawOne = densityTrace(one, numQubits, dd);
  constexpr double tolerance = 1e-10;
  if (!std::isfinite(rawZero) || !std::isfinite(rawOne) ||
      rawZero < -tolerance || rawOne < -tolerance) {
    return diagnosticOp->emitError()
           << "density matrix has invalid measurement probabilities";
  }
  const double pzero = std::max(0.0, rawZero);
  const double pone = std::max(0.0, rawOne);
  const double sum = pzero + pone;
  if (!std::isfinite(sum) || std::abs(sum - 1.0) > tolerance) {
    return diagnosticOp->emitError()
           << "density matrix must have unit trace for measurement";
  }

  std::uniform_real_distribution<double> distribution(0.0, sum);
  const bool measuredOne = distribution(rng) >= pzero;
  auto collapsed = measuredOne ? one : zero;
  const double probability = measuredOne ? pone : pzero;
  if (!(probability > 0.0)) {
    return diagnosticOp->emitError()
           << "density measurement selected a zero-probability outcome";
  }
  collapsed.w =
      dd.cn.lookup(static_cast<dd::ComplexValue>(collapsed.w) / probability);
  dd.incRef(collapsed);
  dd.decRef(state.matrix);
  state.matrix = collapsed;
  dd.garbageCollect();
  return measuredOne ? '1' : '0';
}

static FailureOr<char> measureState(dd::VectorDD& state, const qc::Qubit wire,
                                    const size_t /*numQubits*/, dd::Package& dd,
                                    std::mt19937_64& rng,
                                    Operation* /*diagnosticOp*/) {
  return dd.measureOneCollapsing(state, wire, rng);
}

static FailureOr<char> measureState(DensityState& state, const qc::Qubit wire,
                                    const size_t numQubits, dd::Package& dd,
                                    std::mt19937_64& rng,
                                    Operation* diagnosticOp) {
  return measureDensity(state, wire, numQubits, dd, rng, diagnosticOp);
}

static FailureOr<std::string>
measureAllDensity(DensityState& state, const size_t numQubits, dd::Package& dd,
                  std::mt19937_64& rng, Operation* diagnosticOp) {
  std::string result(numQubits, '0');
  for (size_t i = numQubits; i > 0; --i) {
    const auto wire = static_cast<qc::Qubit>(i - 1);
    auto measured =
        measureDensity(state, wire, numQubits, dd, rng, diagnosticOp);
    if (failed(measured)) {
      return failure();
    }
    result[numQubits - i] = *measured;
  }
  return result;
}

template <typename StateDD>
static LogicalResult applyOp(Operation& op, WalkState& walk, StateDD& state) {
  return TypeSwitch<Operation*, LogicalResult>(&op)
      .template Case<StaticOp, SinkOp>([](auto) { return success(); })
      .template Case<arith::ConstantOp>([&](arith::ConstantOp constant) {
        return recordConstant(constant, *walk.classical);
      })
      .template Case<AllocOp>([&](AllocOp alloc) -> LogicalResult {
        if constexpr (std::is_same_v<StateDD, dd::MatrixDD>) {
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
            if constexpr (std::is_same_v<StateDD, dd::MatrixDD>) {
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
            const auto* tracked = walk.tensors->lookup(dealloc.getTensor());
            if (tracked == nullptr) {
              return dealloc.emitError()
                     << "qtensor is not mapped for QCO DD simulation";
            }
            TensorSlots slots = *tracked;
            walk.tensors->erase(dealloc.getTensor());
            const bool containsDeferredMeasurement =
                llvm::any_of(slots, [&](const std::optional<qc::Qubit> wire) {
                  return wire &&
                         llvm::any_of(walk.classical->deferredMeasurements,
                                      [&](const auto& measurement) {
                                        return measurement.second == *wire;
                                      });
                });
            if (walk.deallocationMode == DeallocationMode::PreserveAll ||
                (walk.deallocationMode == DeallocationMode::PreserveDeferred &&
                 containsDeferredMeasurement)) {
              return success();
            }
            if constexpr (!std::is_same_v<StateDD, dd::MatrixDD>) {
              SmallVector<qc::Qubit> wires;
              for (const auto wire : slots) {
                if (wire) {
                  wires.push_back(*wire);
                }
              }
              llvm::sort(wires, [](qc::Qubit lhs, qc::Qubit rhs) {
                return lhs > rhs;
              });
              for (const qc::Qubit wire : wires) {
                if (failed(deallocateWire(wire, walk, state, dealloc))) {
                  return failure();
                }
              }
            }
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
          arith::RemUIOp, arith::RemSIOp, arith::MaxSIOp, arith::MinSIOp,
          arith::MaxUIOp, arith::MinUIOp, arith::ShLIOp, arith::ShRUIOp,
          arith::ShRSIOp, arith::CmpIOp, arith::SelectOp, arith::ExtUIOp,
          arith::ExtSIOp, arith::IndexCastUIOp, arith::IndexCastOp,
          arith::TruncIOp, arith::AddFOp, arith::SubFOp, arith::MulFOp,
          arith::DivFOp, arith::RemFOp, arith::MaximumFOp, arith::MinimumFOp,
          arith::MaxNumFOp, arith::MinNumFOp, arith::NegFOp, arith::CmpFOp,
          arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp,
          math::AbsFOp, math::CeilOp, math::CosOp, math::ExpOp, math::FloorOp,
          math::LogOp, math::SinOp, math::SqrtOp, math::TanOp, math::PowFOp>(
          [&](Operation* classicalOp) {
            return applyClassicalOp(*classicalOp, *walk.classical);
          })
      .template Case<func::ReturnOp>([&](func::ReturnOp returnOp) {
        return validateReturn(returnOp, *walk.qubits, *walk.tensors);
      })
      .template Case<MeasureOp>([&](MeasureOp measureOp) -> LogicalResult {
        if constexpr (std::is_same_v<StateDD, dd::MatrixDD>) {
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
          auto measured = measureState(state, *q, walk.qubits->numQubits,
                                       *walk.dd, *walk.rng, measureOp);
          if (failed(measured)) {
            return failure();
          }
          const char bit = *measured;
          walk.classical->values[measureOp.getResult()] = bit == '1';
          walk.qubits->bind(measureOp.getQubitOut(), *q);
          return success();
        }
      })
      .template Case<ResetOp>([&](ResetOp resetOp) -> LogicalResult {
        if constexpr (std::is_same_v<StateDD, dd::MatrixDD>) {
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
          auto measured = measureState(state, *q, walk.qubits->numQubits,
                                       *walk.dd, *walk.rng, resetOp);
          if (failed(measured)) {
            return failure();
          }
          const char bit = *measured;
          if (bit == '1') {
            applyStateOperation(
                walk.dd->makeGateDD(
                    dd::opToSingleQubitGateMatrix(qc::OpType::X), *q),
                *walk.dd, state);
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

        if (failed(bindValuePairs(call.getArgOperands(), callee.getArguments(),
                                  walk, call))) {
          return failure();
        }

        auto returnOp = walkFunctionBody(callee, walk, state);
        if (failed(returnOp)) {
          return failure();
        }

        // Map callee return operands onto call results via the return op.
        return bindValuePairs(returnOp->getOperands(), call.getResults(), walk,
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
static FailureOr<func::ReturnOp>
walkFunctionBody(func::FuncOp func, WalkState& walk, StateDD& state) {
  auto terminator = walkConcreteCFG(
      func.getBody().front(), walk, state,
      [](Operation* op) { return isa<func::ReturnOp>(op); }, "function");
  if (failed(terminator)) {
    return failure();
  }
  return cast<func::ReturnOp>(*terminator);
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
};
} // namespace

static FailureOr<PreparedState>
prepare(func::FuncOp func, const dd::Package& dd, const DDBindings& bindings,
        const bool bindEntryAllocations = false) {
  if (func.isDeclaration()) {
    return func.emitError() << "QCO DD construction requires a function body";
  }

  PreparedState prepared;
  QubitMap& qubits = prepared.qubits;
  for (Block& block : func.getBody()) {
    for (StaticOp staticOp : block.getOps<StaticOp>()) {
      const auto index = static_cast<size_t>(staticOp.getIndex());
      if (index >= dd::Package::MAX_POSSIBLE_QUBITS) {
        return staticOp.emitError()
               << "static qubit index exceeds the supported qubit range";
      }
      const auto q = static_cast<qc::Qubit>(index);
      qubits.bind(staticOp.getQubit(), q);
      qubits.numQubits = std::max(qubits.numQubits, static_cast<size_t>(q) + 1);
    }
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

static bool mayAllocateQubits(func::FuncOp func, DenseSet<Operation*>& active) {
  if (!active.insert(func).second) {
    return true;
  }
  const auto activeGuard = llvm::make_scope_exit([&] { active.erase(func); });

  bool allocation = false;
  func.walk([&](Operation* op) {
    if (allocation || isa<AllocOp, qtensor::AllocOp>(op)) {
      allocation = true;
      return;
    }
    auto call = dyn_cast<func::CallOp>(op);
    if (!call) {
      return;
    }
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    allocation =
        !callee || callee.isDeclaration() || mayAllocateQubits(callee, active);
  });
  return allocation;
}

FailureOr<size_t> getNumQubits(func::FuncOp func) {
  if (func.isDeclaration()) {
    return func.emitError() << "QCO DD capacity requires a function body";
  }

  constexpr size_t maxQubits = dd::Package::MAX_POSSIBLE_QUBITS;
  size_t required = 0;
  LogicalResult result = success();
  const auto addQubits = [&](const size_t count, Operation* op) {
    if (failed(result)) {
      return;
    }
    if (count > maxQubits - required) {
      result = op->emitError()
               << "QCO function exceeds the supported qubit range";
      return;
    }
    required += count;
  };

  func.walk([&](StaticOp staticOp) {
    if (failed(result)) {
      return;
    }
    const auto index = static_cast<size_t>(staticOp.getIndex());
    if (index >= maxQubits) {
      result = staticOp.emitError()
               << "static qubit index exceeds the supported qubit range";
      return;
    }
    required = std::max(required, index + 1U);
  });

  if (required == 0) {
    for (Value arg : func.getArguments()) {
      if (isa<QubitType>(arg.getType())) {
        addQubits(1, func);
        continue;
      }
      if (!isQTensorType(arg.getType())) {
        continue;
      }
      const auto type = cast<RankedTensorType>(arg.getType());
      if (type.isDynamicDim(0)) {
        return func.emitError()
               << "dynamic qtensor arguments prevent static DD capacity "
                  "calculation";
      }
      addQubits(static_cast<size_t>(type.getDimSize(0)), func);
    }
  }

  Block* entry = &func.getBody().front();
  func.walk([&](AllocOp alloc) {
    if (alloc->getBlock() != entry) {
      result = alloc.emitError()
               << "quantum allocations outside the entry block prevent "
                  "static DD capacity calculation";
      return;
    }
    addQubits(1, alloc);
  });
  func.walk([&](qtensor::AllocOp alloc) {
    if (failed(result)) {
      return;
    }
    if (alloc->getBlock() != entry) {
      result = alloc.emitError()
               << "quantum allocations outside the entry block prevent "
                  "static DD capacity calculation";
      return;
    }
    const auto attr = mqt::valueToConstantAttr(alloc.getSize());
    const auto integer = attr ? dyn_cast<IntegerAttr>(*attr) : IntegerAttr{};
    if (!integer) {
      result = alloc.emitError()
               << "dynamic qtensor allocation prevents static DD capacity "
                  "calculation";
      return;
    }
    const int64_t count = integer.getInt();
    if (count <= 0) {
      result = alloc.emitError()
               << "qtensor allocation size must be a positive integer";
      return;
    }
    addQubits(static_cast<size_t>(count), alloc);
  });

  DenseSet<Operation*> active;
  func.walk([&](func::CallOp call) {
    if (failed(result)) {
      return;
    }
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if (!callee || callee.isDeclaration() ||
        mayAllocateQubits(callee, active)) {
      result = call.emitError()
               << "calls that may allocate qubits prevent static DD capacity "
                  "calculation";
    }
  });

  if (failed(result)) {
    return failure();
  }
  return required;
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
             const DDBindings& bindings,
             const DenseSet<Operation*>* deferredMeasurements = nullptr,
             ClassicalEnv* finalClassical = nullptr,
             const DeallocationMode deallocationMode = DeallocationMode::Apply,
             const bool validateQuantumReturn = true) {
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
                      .deferredMeasurements = deferredMeasurements,
                      .deallocationMode = deallocationMode};
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
                                 dd::Package& dd, const DDBindings& bindings) {
  auto prepared = prepare(func, dd, bindings);
  if (failed(prepared)) {
    dd.decRef(in);
    return failure();
  }
  return simulateImpl(func, in, dd, *prepared, nullptr, bindings);
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

dd::MatrixDD makeDensityMatrix(const dd::VectorDD& state,
                               const size_t numQubits, dd::Package& dd) {
  if (numQubits > dd.qubits()) {
    throw std::invalid_argument(
        "numQubits exceeds the capacity of the DD package");
  }
  if (!state.isTerminal() && std::cmp_greater_equal(state.p->v, numQubits)) {
    throw std::invalid_argument(
        "numQubits does not cover the input state's highest qubit");
  }
  std::map<std::tuple<dd::vNode*, dd::vNode*, int64_t>, dd::mCachedEdge> cache;
  const auto root = buildDensityMatrix(
      state, state, static_cast<int64_t>(numQubits) - 1, dd, cache);
  const auto density = dd::MatrixDD{.p = root.p, .w = dd.cn.lookup(root.w)};
  dd.incRef(density);
  return density;
}

namespace {
struct DensitySimulationResult {
  dd::MatrixDD matrix;
  size_t numQubits;
};
} // namespace

static FailureOr<DensitySimulationResult>
simulateDensityImpl(func::FuncOp func, const dd::MatrixDD& in, dd::Package& dd,
                    const PreparedState& prepared, std::mt19937_64* rng,
                    const DDBindings& bindings,
                    const DenseSet<Operation*>* deferredMeasurements = nullptr,
                    ClassicalEnv* finalClassical = nullptr) {
  QubitMap qubits = prepared.qubits;
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

  DensityState state{in};
  if (failed(walkFunction(func, walkState, state))) {
    dd.decRef(state.matrix);
    return failure();
  }
  if (finalClassical != nullptr) {
    *finalClassical = std::move(classical);
  }
  return DensitySimulationResult{.matrix = state.matrix,
                                 .numQubits = qubits.numQubits};
}

FailureOr<dd::MatrixDD> simulateDensity(func::FuncOp func,
                                        const dd::MatrixDD& in, dd::Package& dd,
                                        const DDBindings& bindings) {
  auto prepared = prepare(func, dd, bindings);
  if (failed(prepared)) {
    dd.decRef(in);
    return failure();
  }
  auto result = simulateDensityImpl(func, in, dd, *prepared, nullptr, bindings);
  if (failed(result)) {
    return failure();
  }
  return result->matrix;
}

FailureOr<dd::MatrixDD> simulateDensity(func::FuncOp func,
                                        const dd::MatrixDD& in, dd::Package& dd,
                                        std::mt19937_64& rng,
                                        const DDBindings& bindings) {
  auto prepared = prepare(func, dd, bindings);
  if (failed(prepared)) {
    dd.decRef(in);
    return failure();
  }
  auto result = simulateDensityImpl(func, in, dd, *prepared, &rng, bindings);
  if (failed(result)) {
    return failure();
  }
  return result->matrix;
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
                                           ArrayRef<Value> outputs,
                                           const bool allowUnusedResult) {
  Value result = measure.getResult();
  if (result.use_empty()) {
    return allowUnusedResult;
  }
  return llvm::all_of(result.getUses(), [&](const OpOperand& use) {
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
                                       const bool allowUnusedResult,
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
             hasOutputOnlyMeasurementResult(measure, outputs,
                                            allowUnusedResult) &&
             hasOnlyTerminalQuantumUses(measure.getQubitOut(), std::nullopt,
                                        func, outputs, allowUnusedResult,
                                        visited);
    }
    if (auto fromElements = dyn_cast<qtensor::FromElementsOp>(owner)) {
      return !tensorSlot &&
             hasOnlyTerminalQuantumUses(fromElements.getResult(),
                                        use.getOperandNumber(), func, outputs,
                                        allowUnusedResult, visited);
    }
    if (auto insert = dyn_cast<qtensor::InsertOp>(owner)) {
      const auto index = getConstantTensorIndex(insert.getIndex());
      if (!index) {
        return false;
      }
      if (use.get() == insert.getScalar()) {
        return !tensorSlot &&
               hasOnlyTerminalQuantumUses(insert.getResult(), index, func,
                                          outputs, allowUnusedResult, visited);
      }
      return tensorSlot && *tensorSlot != *index &&
             hasOnlyTerminalQuantumUses(insert.getResult(), tensorSlot, func,
                                        outputs, allowUnusedResult, visited);
    }
    if (auto extract = dyn_cast<qtensor::ExtractOp>(owner)) {
      const auto index = getConstantTensorIndex(extract.getIndex());
      if (!tensorSlot || !index) {
        return false;
      }
      if (*tensorSlot == *index) {
        return hasOnlyTerminalQuantumUses(extract.getResult(), std::nullopt,
                                          func, outputs, allowUnusedResult,
                                          visited);
      }
      return hasOnlyTerminalQuantumUses(extract.getOutTensor(), tensorSlot,
                                        func, outputs, allowUnusedResult,
                                        visited);
    }
    return false;
  });
}

static bool isDeferrableMeasurement(MeasureOp measure, func::FuncOp func,
                                    ArrayRef<Value> outputs,
                                    const bool allowUnusedResult) {
  if (!hasOutputOnlyMeasurementResult(measure, outputs, allowUnusedResult)) {
    return false;
  }
  DenseSet<Value> visited;
  return hasOnlyTerminalQuantumUses(measure.getQubitOut(), std::nullopt, func,
                                    outputs, allowUnusedResult, visited);
}

static void analyzeSampling(func::FuncOp func, ArrayRef<Value> outputs,
                            DenseSet<Operation*>& active, SamplingPlan& plan,
                            const bool allowUnusedMeasurementResults) {
  Operation* funcOp = func.getOperation();
  if (!active.insert(funcOp).second) {
    plan.dynamic = true;
    return;
  }
  func.getBody().walk([&](Operation* op) {
    if (isa<ResetOp>(op)) {
      plan.dynamic = true;
    } else if (auto measure = dyn_cast<MeasureOp>(op)) {
      if (isDeferrableMeasurement(measure, func, outputs,
                                  allowUnusedMeasurementResults)) {
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
        // A callee return is not terminal for the entry function. Without
        // following call-result dataflow, unused measurement results in
        // callees must remain collapsing.
        analyzeSampling(callee, outputs, active, plan,
                        /*allowUnusedMeasurementResults=*/false);
      }
    }
  });
  active.erase(funcOp);
}

static FailureOr<SamplingPlan>
getSamplingPlan(func::FuncOp func, const bool validateHistogramOutputs = true,
                const bool allowUnusedMeasurementResults = false) {
  Block& entry = func.getBody().front();
  SamplingPlan plan;
  if (validateHistogramOutputs && !func.getBody().hasOneBlock()) {
    if (llvm::any_of(func.getFunctionType().getResults(),
                     [](Type type) { return isa<cbit::RegisterType>(type); })) {
      return func.emitError()
             << "QCO DD sampling does not support CBit results from "
                "multi-block functions";
    }
    plan.dynamic = true;
    return plan;
  }

  if (validateHistogramOutputs && !isa<func::ReturnOp>(entry.getTerminator())) {
    return func.emitError()
           << "single-block QCO DD sampling requires func.return";
  }

  bool hasOther = false;
  func.getBody().walk([&](func::ReturnOp returnOp) {
    for (Value value : returnOp.getOperands()) {
      if (isa<cbit::RegisterType>(value.getType())) {
        plan.outputs.push_back(value);
      } else {
        hasOther = true;
      }
    }
  });
  if (validateHistogramOutputs && !plan.outputs.empty() && hasOther) {
    return entry.getTerminator()->emitError()
           << "QCO DD sampling does not support mixed CBit and non-CBit "
              "results";
  }

  DenseSet<Operation*> active;
  analyzeSampling(func, plan.outputs, active, plan,
                  allowUnusedMeasurementResults);
  return plan;
}

FailureOr<dd::VectorDD> simulateStatevector(func::FuncOp func, dd::Package& dd,
                                            std::mt19937_64& rng) {
  const auto requiredQubits = getNumQubits(func);
  if (failed(requiredQubits)) {
    return failure();
  }
  if (dd.qubits() < *requiredQubits) {
    return func.emitError()
           << "DD package has " << dd.qubits()
           << " qubits but function requires " << *requiredQubits;
  }
  const DDBindings bindings;
  auto prepared = prepare(func, dd, bindings);
  if (failed(prepared)) {
    return failure();
  }
  auto plan = getSamplingPlan(func, /*validateHistogramOutputs=*/false,
                              /*allowUnusedMeasurementResults=*/true);
  if (failed(plan)) {
    return failure();
  }
  return simulateImpl(func, dd::makeZeroState(prepared->qubits.numQubits, dd),
                      dd, *prepared, &rng, bindings,
                      &plan->deferredMeasurements, nullptr,
                      DeallocationMode::PreserveAll,
                      /*validateQuantumReturn=*/false);
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
           const size_t shots, std::mt19937_64& rng, const DDBindings& bindings,
           const bool preserveDeallocatedQubits = false) {
  const auto inputGuard = llvm::make_scope_exit([&] { dd.decRef(in); });
  auto prepared = prepare(func, dd, bindings);
  if (failed(prepared)) {
    return failure();
  }
  auto plan = getSamplingPlan(func);
  if (failed(plan)) {
    return failure();
  }

  const size_t inputQubits =
      in.isTerminal() ? 0U : static_cast<size_t>(in.p->v) + 1U;
  if (inputQubits < prepared->qubits.numQubits) {
    return func.emitError()
           << "input state has " << inputQubits << " qubits but function uses "
           << prepared->qubits.numQubits;
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
    dd.incRef(in);
    const auto deallocationMode =
        preserveDeallocatedQubits || !plan->outputs.empty()
            ? DeallocationMode::PreserveAll
            : DeallocationMode::PreserveDeferred;
    auto state =
        simulateImpl(func, in, dd, *prepared, nullptr, bindings,
                     &plan->deferredMeasurements, &classical, deallocationMode,
                     /*validateQuantumReturn=*/!preserveDeallocatedQubits);
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
    dd.incRef(in);
    auto state = simulateImpl(
        func, in, dd, *prepared, &rng, bindings, nullptr, &classical,
        preserveDeallocatedQubits || !plan->outputs.empty()
            ? DeallocationMode::PreserveAll
            : DeallocationMode::Apply,
        /*validateQuantumReturn=*/!preserveDeallocatedQubits);
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
sampleDensity(func::FuncOp func, const dd::MatrixDD& in, dd::Package& dd,
              const size_t shots, std::mt19937_64& rng,
              const DDBindings& bindings) {
  const auto inputGuard = llvm::make_scope_exit([&] { dd.decRef(in); });
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

  const auto record = [&](const ClassicalEnv& classical,
                          const StringRef basis) -> LogicalResult {
    auto outcome = encodeOutcome(plan->outputs, classical, basis);
    if (failed(outcome)) {
      return failure();
    }
    ++counts[*outcome];
    return success();
  };

  if (!plan->dynamic) {
    ClassicalEnv classical;
    dd.incRef(in);
    auto simulated =
        simulateDensityImpl(func, in, dd, *prepared, nullptr, bindings,
                            &plan->deferredMeasurements, &classical);
    if (failed(simulated)) {
      return failure();
    }
    const auto simulatedGuard =
        llvm::make_scope_exit([&] { dd.decRef(simulated->matrix); });
    for (size_t i = 0; i < shots; ++i) {
      dd.incRef(simulated->matrix);
      DensityState sampleState{simulated->matrix};
      const auto sampleGuard =
          llvm::make_scope_exit([&] { dd.decRef(sampleState.matrix); });
      auto outcome = measureAllDensity(sampleState, simulated->numQubits, dd,
                                       rng, func.getOperation());
      if (failed(outcome)) {
        return failure();
      }
      if (failed(record(classical, *outcome))) {
        return failure();
      }
    }
    return counts;
  }

  for (size_t i = 0; i < shots; ++i) {
    ClassicalEnv classical;
    dd.incRef(in);
    auto simulated = simulateDensityImpl(func, in, dd, *prepared, &rng,
                                         bindings, nullptr, &classical);
    if (failed(simulated)) {
      return failure();
    }
    DensityState sampleState{simulated->matrix};
    const auto sampleGuard =
        llvm::make_scope_exit([&] { dd.decRef(sampleState.matrix); });
    std::string basis;
    if (plan->outputs.empty()) {
      auto measured = measureAllDensity(sampleState, simulated->numQubits, dd,
                                        rng, func.getOperation());
      if (failed(measured)) {
        return failure();
      }
      basis = std::move(*measured);
    }
    if (failed(record(classical, basis))) {
      return failure();
    }
  }
  return counts;
}

FailureOr<std::map<std::string, size_t>>
sample(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd,
       const size_t shots, std::mt19937_64& rng, const DDBindings& bindings) {
  return sampleImpl(func, in, dd, shots, rng, bindings);
}

FailureOr<std::map<std::string, size_t>>
sample(func::FuncOp func, dd::Package& dd, const size_t shots,
       std::mt19937_64& rng, const DDBindings& bindings) {
  auto prepared = prepare(func, dd, bindings);
  if (failed(prepared)) {
    return failure();
  }
  return sampleImpl(func, dd::makeZeroState(prepared->qubits.numQubits, dd), dd,
                    shots, rng, bindings);
}

FailureOr<std::map<std::string, size_t>>
sampleAllQubits(func::FuncOp func, dd::Package& dd, const size_t shots,
                std::mt19937_64& rng, const DDBindings& bindings) {
  auto prepared = prepare(func, dd, bindings);
  if (failed(prepared)) {
    return failure();
  }
  return sampleImpl(func, dd::makeZeroState(prepared->qubits.numQubits, dd), dd,
                    shots, rng, bindings,
                    /*preserveDeallocatedQubits=*/true);
}

} // namespace mlir::qco
