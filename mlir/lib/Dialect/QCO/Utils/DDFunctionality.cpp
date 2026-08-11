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

#include "dd/DDDefinitions.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/Definitions.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>

#include <algorithm>
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

constexpr int64_t MAX_CONTROL_FLOW_TRIPS = 10000;

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
          wire = std::nullopt;
        } else if (*wire > released) {
          --*wire;
        }
      }
    }
  }
};

[[nodiscard]] static bool isQTensorType(Type type) {
  const auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getRank() == 1 &&
         isa<QubitType>(tensorType.getElementType());
}

struct ClassicalEnv {
  DenseMap<Value, bool> bools;
  DenseMap<Value, int64_t> indices;
  DenseMap<Value, APInt> integers;
  DenseMap<Value, double> floats;
  using Scalar = std::variant<bool, int64_t, APInt, double>;
  struct MemRefStorage {
    SmallVector<Scalar> values;
    bool live = true;
  };
  /// Backing storage for one-dimensional classical registers.
  DenseMap<Value, std::shared_ptr<MemRefStorage>> memrefs;

  LogicalResult bindFrom(Value source, Value dest, Operation* op) {
    if (dest.getType().isInteger(1)) {
      const auto it = bools.find(source);
      if (it == bools.end()) {
        return op->emitError()
               << "classical i1 SSA value is not mapped for QCO DD simulation";
      }
      bools[dest] = it->second;
      return success();
    }
    if (isa<IndexType>(dest.getType())) {
      const auto it = indices.find(source);
      if (it == indices.end()) {
        return op->emitError() << "classical index SSA value is not mapped "
                                  "for QCO DD simulation";
      }
      indices[dest] = it->second;
      return success();
    }
    if (isa<IntegerType>(dest.getType())) {
      const auto it = integers.find(source);
      if (it == integers.end()) {
        return op->emitError()
               << "classical integer SSA value is not mapped for QCO DD "
                  "simulation";
      }
      integers[dest] = it->second;
      return success();
    }
    if (isa<FloatType>(dest.getType())) {
      const auto it = floats.find(source);
      if (it == floats.end()) {
        return op->emitError()
               << "classical floating-point SSA value is not mapped for QCO "
                  "DD simulation";
      }
      floats[dest] = it->second;
      return success();
    }
    if (isa<MemRefType>(dest.getType())) {
      const auto it = memrefs.find(source);
      if (it == memrefs.end() || !it->second->live) {
        return op->emitError()
               << "classical memref is not mapped for QCO DD simulation";
      }
      memrefs[dest] = it->second;
      return success();
    }
    return op->emitError()
           << "unsupported classical type for QCO DD simulation: "
           << dest.getType();
  }
};

struct DecodedGate {
  qc::OpType type = qc::OpType::None;
  std::vector<dd::fp> params;
};

struct WalkState {
  // Non-owning handles into the active simulation frame.
  QubitMap& qubits; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
  TensorMap&
      tensors; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
  ClassicalEnv&
      classical;   // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
  dd::Package& dd; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
  std::mt19937_64* rng = nullptr;
  std::string* classicalBits = nullptr;
  DenseSet<Operation*>* activeCalls = nullptr;
};

/// Distinguishes a density operator from the matrix used for functionality
/// construction while retaining the DD package's matrix representation.
struct DensityState {
  dd::MatrixDD matrix;
};

/// Erases @p op from @p set on destruction (used around `func.call`).
struct ActiveCallGuard {
  DenseSet<Operation*>* set = nullptr;
  Operation* op = nullptr;

  ActiveCallGuard(DenseSet<Operation*>* activeSet, Operation* callee)
      : set(activeSet), op(callee) {}
  ~ActiveCallGuard() {
    if (set != nullptr && op != nullptr) {
      set->erase(op);
    }
  }
  ActiveCallGuard(const ActiveCallGuard&) = delete;
  ActiveCallGuard& operator=(const ActiveCallGuard&) = delete;
};

} // namespace

static FailureOr<double>
resolveDouble(Value value, const ClassicalEnv& classical, Operation* op) {
  if (const auto it = classical.floats.find(value);
      it != classical.floats.end()) {
    return it->second;
  }
  if (const auto constant = utils::valueToDouble(value)) {
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
  const qc::OpType type =
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
                     const size_t numOperands, const int64_t level,
                     const size_t row, const size_t col) {
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
                                        const size_t numQubits,
                                        const ArrayRef<qc::Qubit> wires) {
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
    if (!edge.isTerminal() && edge.p->v == level) {
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
    auto theta = resolveDouble(gphase.getTheta(), walk.classical, op);
    if (failed(theta)) {
      return failure();
    }
    if constexpr (!std::is_same_v<StateDD, DensityState>) {
      auto id = dd::Package::makeIdent();
      id.w = walk.dd.cn.lookup(std::cos(*theta), std::sin(*theta));
      applyStateOperation(id, walk.dd, state);
    }
    return success();
  }
  if (isa<BarrierOp>(op)) {
    return walk.qubits.remapUnitary(unitary);
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

  auto wiresOr = walk.qubits.lookupRange(unitary.getInputQubits(), op);
  if (failed(wiresOr)) {
    return failure();
  }
  const ArrayRef<qc::Qubit> wires = *wiresOr;
  if (wires.size() >= 63 ||
      local.rows() != static_cast<int64_t>(size_t{1} << wires.size())) {
    return unitary.emitError()
           << "unitary matrix dimension does not match its target count";
  }

  if (wires.size() == 1) {
    const dd::GateMatrix mat{local(0, 0), local(0, 1), local(1, 0),
                             local(1, 1)};
    applyStateOperation(walk.dd.makeGateDD(mat, wires[0]), walk.dd, state);
    return walk.qubits.remapUnitary(unitary);
  }

  if (wires.size() == 2) {
    dd::TwoQubitGateMatrix mat{};
    for (size_t row = 0; row < mat.size(); ++row) {
      for (size_t col = 0; col < mat[row].size(); ++col) {
        mat[row][col] =
            local(static_cast<int64_t>(row), static_cast<int64_t>(col));
      }
    }
    applyStateOperation(walk.dd.makeTwoQubitGateDD(mat, wires[0], wires[1]),
                        walk.dd, state);
    return walk.qubits.remapUnitary(unitary);
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
        walk.dd.makeThreeQubitGateDD(mat, wires[0], wires[1], wires[2]),
        walk.dd, state);
    return walk.qubits.remapUnitary(unitary);
  }

  applyStateOperation(
      makeEmbeddedLocalDD(walk.dd, local, walk.qubits.numQubits, wires),
      walk.dd, state);
  return walk.qubits.remapUnitary(unitary);
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
  auto targets = walk.qubits.lookupRange(targetVals, unitary.getOperation());
  if (failed(targets)) {
    return failure();
  }
  applyStateOperation(
      getStandardOperationDD(walk.dd, gate.type, gate.params, controls,
                             {targets->begin(), targets->end()}),
      walk.dd, state);
  return walk.qubits.remapUnitary(unitary);
}

static LogicalResult validateReturn(func::ReturnOp returnOp,
                                    const QubitMap& qubits,
                                    const TensorMap& tensors) {
  qc::Qubit expected = 0;
  for (Value value : returnOp.getOperands()) {
    if (isa<QubitType>(value.getType())) {
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
      continue;
    }
    if (!isQTensorType(value.getType())) {
      continue;
    }
    const auto* slots = tensors.lookup(value);
    if (slots == nullptr) {
      return returnOp.emitError()
             << "returned qtensor SSA value is not mapped for QCO DD "
                "simulation";
    }
    for (const auto wire : *slots) {
      if (!wire) {
        return returnOp.emitError()
               << "returned qtensor contains an extracted element";
      }
      if (*wire != expected) {
        return returnOp.emitError()
               << "returned qubits must preserve canonical wire order; qubit "
                  "result "
               << static_cast<size_t>(expected) << " maps to wire "
               << static_cast<size_t>(*wire);
      }
      ++expected;
    }
  }
  return success();
}

static LogicalResult recordConstant(arith::ConstantOp constant,
                                    ClassicalEnv& classical) {
  // `arith.constant true/false` is a BoolAttr; other integers are IntegerAttr.
  if (auto boolAttr = dyn_cast<BoolAttr>(constant.getValue())) {
    classical.bools[constant.getResult()] = boolAttr.getValue();
    return success();
  }
  if (auto floatAttr = dyn_cast<FloatAttr>(constant.getValue())) {
    classical.floats[constant.getResult()] =
        floatAttr.getValue().convertToDouble();
    return success();
  }
  auto attr = dyn_cast<IntegerAttr>(constant.getValue());
  if (!attr) {
    return success();
  }
  if (constant.getType().isInteger(1)) {
    classical.bools[constant.getResult()] = attr.getValue() != 0;
  } else if (isa<IndexType>(constant.getType())) {
    classical.indices[constant.getResult()] = attr.getInt();
  } else if (isa<IntegerType>(constant.getType())) {
    classical.integers[constant.getResult()] = attr.getValue();
  }
  return success();
}

static LogicalResult applyBindings(func::FuncOp func,
                                   const DDBindings& bindings,
                                   ClassicalEnv& classical) {
  for (const auto& [value, attr] : bindings) {
    const auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &func.getBody().front()) {
      return func.emitError()
             << "QCO DD bindings must target entry-block arguments";
    }
    Type type = value.getType();
    if (isQTensorType(type)) {
      if (cast<RankedTensorType>(type).isDynamicDim(0) &&
          isa<IntegerAttr>(attr)) {
        continue;
      }
      return func.emitError()
             << "QCO DD qtensor bindings require a dynamic qtensor argument "
                "and an integer extent";
    }
    if (type.isInteger(1)) {
      if (auto boolean = dyn_cast<BoolAttr>(attr)) {
        classical.bools[value] = boolean.getValue();
        continue;
      }
      if (auto integer = dyn_cast<IntegerAttr>(attr)) {
        classical.bools[value] = integer.getValue() != 0;
        continue;
      }
    } else if (isa<IndexType>(type)) {
      if (auto integer = dyn_cast<IntegerAttr>(attr)) {
        classical.indices[value] = integer.getInt();
        continue;
      }
    } else if (auto integerType = dyn_cast<IntegerType>(type)) {
      if (auto integer = dyn_cast<IntegerAttr>(attr)) {
        classical.integers[value] =
            integer.getValue().sextOrTrunc(integerType.getWidth());
        continue;
      }
    } else if (isa<FloatType>(type)) {
      if (auto floating = dyn_cast<FloatAttr>(attr)) {
        classical.floats[value] = floating.getValue().convertToDouble();
        continue;
      }
    }
    return func.emitError() << "QCO DD binding attribute " << attr
                            << " does not match argument type " << type;
  }
  return success();
}

static FailureOr<bool> lookupBool(Value value, ClassicalEnv& classical,
                                  Operation* op) {
  const auto it = classical.bools.find(value);
  if (it == classical.bools.end()) {
    return op->emitError()
           << "classical i1 SSA value is not mapped for QCO DD simulation";
  }
  return it->second;
}

static FailureOr<int64_t> lookupIndex(Value value, ClassicalEnv& classical,
                                      Operation* op) {
  const auto it = classical.indices.find(value);
  if (it == classical.indices.end()) {
    return op->emitError()
           << "classical index SSA value is not mapped for QCO DD simulation";
  }
  return it->second;
}

static FailureOr<APInt> lookupInteger(Value value, ClassicalEnv& classical,
                                      Operation* op) {
  const auto it = classical.integers.find(value);
  if (it == classical.integers.end()) {
    return op->emitError()
           << "classical integer SSA value is not mapped for QCO DD simulation";
  }
  return it->second;
}

static FailureOr<double> lookupFloat(Value value, ClassicalEnv& classical,
                                     Operation* op) {
  const auto it = classical.floats.find(value);
  if (it == classical.floats.end()) {
    return op->emitError()
           << "classical floating-point SSA value is not mapped "
              "for QCO DD simulation";
  }
  return it->second;
}

static FailureOr<TensorSlots> allocateZeroQubits(const size_t count,
                                                 WalkState& walk,
                                                 dd::VectorDD& state,
                                                 Operation* op) {
  if (count == 0) {
    return op->emitError()
           << "quantum allocation size must be positive for QCO DD simulation";
  }
  if (walk.qubits.numQubits > walk.dd.qubits() ||
      count > walk.dd.qubits() - walk.qubits.numQubits) {
    return op->emitError() << "DD package has " << walk.dd.qubits()
                           << " qubits but allocation requires "
                           << walk.qubits.numQubits + count;
  }

  const size_t first = walk.qubits.numQubits;
  auto zeros = dd::makeZeroState(count, walk.dd, first);
  auto extended = walk.dd.kronecker(zeros, state, first, /*incIdx=*/false);
  walk.dd.incRef(extended);
  walk.dd.decRef(zeros);
  walk.dd.decRef(state);
  state = extended;

  TensorSlots slots;
  slots.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    slots.emplace_back(static_cast<qc::Qubit>(first + i));
  }
  walk.qubits.numQubits += count;
  return slots;
}

static FailureOr<TensorSlots> allocateZeroQubits(const size_t count,
                                                 WalkState& walk,
                                                 DensityState& state,
                                                 Operation* op) {
  if (count == 0) {
    return op->emitError()
           << "quantum allocation size must be positive for QCO DD simulation";
  }
  if (walk.qubits.numQubits > walk.dd.qubits() ||
      count > walk.dd.qubits() - walk.qubits.numQubits) {
    return op->emitError() << "DD package has " << walk.dd.qubits()
                           << " qubits but allocation requires "
                           << walk.qubits.numQubits + count;
  }

  const size_t first = walk.qubits.numQubits;
  auto extended = state.matrix;
  for (size_t i = 0; i < count; ++i) {
    extended = walk.dd.makeDDNode<dd::mNode, dd::Edge>(
        static_cast<qc::Qubit>(first + i),
        {extended, dd::MatrixDD::zero(), dd::MatrixDD::zero(),
         dd::MatrixDD::zero()});
  }
  walk.dd.incRef(extended);
  walk.dd.decRef(state.matrix);
  state.matrix = extended;
  walk.dd.garbageCollect();

  TensorSlots slots;
  slots.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    slots.emplace_back(static_cast<qc::Qubit>(first + i));
  }
  walk.qubits.numQubits += count;
  return slots;
}

/**
 * @brief Project @p wire onto one basis state and remove its DD level.
 *
 * Nodes above the removed wire are rebuilt with decremented variable indices.
 * A skipped vector-DD level denotes a qubit fixed to zero.
 */
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
  if (wire >= walk.qubits.numQubits) {
    return op->emitError()
           << "deallocated wire is outside the simulated register";
  }
  const auto zero = projectAndRemoveWire(state, wire, false, walk.dd);
  const auto one = projectAndRemoveWire(state, wire, true, walk.dd);
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
                              .w = walk.dd.cn.lookup(norm)};
  walk.dd.incRef(reduced);
  walk.dd.decRef(state);
  state = reduced;
  walk.qubits.releaseWire(wire);
  walk.tensors.releaseWire(wire);
  return success();
}

static LogicalResult deallocateWire(const qc::Qubit wire, WalkState& walk,
                                    DensityState& state, Operation* op) {
  if (wire >= walk.qubits.numQubits) {
    return op->emitError()
           << "deallocated wire is outside the simulated register";
  }
  std::vector<bool> eliminate(walk.qubits.numQubits, false);
  eliminate[wire] = true;
  auto reduced = walk.dd.partialTrace(state.matrix, eliminate);
  // Package::partialTrace uses the normalized matrix trace convention and
  // divides by two per eliminated level. A physical density partial trace does
  // not, so restore that factor here.
  reduced.w = walk.dd.cn.lookup(static_cast<dd::ComplexValue>(reduced.w) * 2.0);
  walk.dd.incRef(reduced);
  walk.dd.decRef(state.matrix);
  state.matrix = reduced;
  walk.dd.garbageCollect();
  walk.qubits.releaseWire(wire);
  walk.tensors.releaseWire(wire);
  return success();
}

static double densityTrace(const dd::MatrixDD& density, const size_t numQubits,
                           dd::Package& dd) {
  const auto normalized = dd.trace(density, numQubits);
  return std::ldexp(normalized.r, static_cast<int>(numQubits));
}

static char measureDensity(DensityState& state, const qc::Qubit wire,
                           const size_t numQubits, dd::Package& dd,
                           std::mt19937_64& rng) {
  const auto project = [&](const dd::GateMatrix& projector) {
    const auto gate = dd.makeGateDD(projector, wire);
    return dd.multiply(dd.multiply(gate, state.matrix), gate);
  };
  auto zero = project(dd::MEAS_ZERO_MAT);
  auto one = project(dd::MEAS_ONE_MAT);
  const double pzero = std::max(0.0, densityTrace(zero, numQubits, dd));
  const double pone = std::max(0.0, densityTrace(one, numQubits, dd));
  const double sum = pzero + pone;
  constexpr double tolerance = 1e-10;
  if (!std::isfinite(sum) || std::abs(sum - 1.0) > tolerance) {
    throw std::runtime_error(
        "density matrix must have unit trace for QCO DD measurement");
  }
  std::uniform_real_distribution<double> distribution(0.0, sum);
  const bool measuredOne = distribution(rng) >= pzero;
  auto collapsed = measuredOne ? one : zero;
  const double probability = measuredOne ? pone : pzero;
  collapsed.w =
      dd.cn.lookup(static_cast<dd::ComplexValue>(collapsed.w) / probability);
  dd.incRef(collapsed);
  dd.decRef(state.matrix);
  state.matrix = collapsed;
  dd.garbageCollect();
  return measuredOne ? '1' : '0';
}

static std::string measureAllDensity(DensityState& state,
                                     const size_t numQubits, dd::Package& dd,
                                     std::mt19937_64& rng) {
  std::string result(numQubits, '0');
  for (size_t i = numQubits; i > 0; --i) {
    const auto wire = static_cast<qc::Qubit>(i - 1);
    result[numQubits - i] = measureDensity(state, wire, numQubits, dd, rng);
  }
  return result;
}

static FailureOr<APInt> lookupIntegerLike(Value value, ClassicalEnv& classical,
                                          Operation* op) {
  if (value.getType().isInteger(1)) {
    auto bit = lookupBool(value, classical, op);
    if (failed(bit)) {
      return failure();
    }
    return APInt(1, *bit);
  }
  if (isa<IndexType>(value.getType())) {
    auto index = lookupIndex(value, classical, op);
    if (failed(index)) {
      return failure();
    }
    return APInt(64, static_cast<uint64_t>(*index));
  }
  return lookupInteger(value, classical, op);
}

static LogicalResult bindIntegerLike(Value out, const APInt& value,
                                     ClassicalEnv& classical) {
  if (out.getType().isInteger(1)) {
    classical.bools[out] = value[0];
  } else if (isa<IndexType>(out.getType())) {
    classical.indices[out] = static_cast<int64_t>(value.getZExtValue());
  } else {
    classical.integers[out] = value;
  }
  return success();
}

static LogicalResult applyIntegerCast(Value in, Value out, Operation* op,
                                      ClassicalEnv& classical,
                                      const bool isSigned) {
  auto value = lookupIntegerLike(in, classical, op);
  if (failed(value)) {
    return failure();
  }
  const unsigned outWidth = isa<IndexType>(out.getType())
                                ? 64U
                                : cast<IntegerType>(out.getType()).getWidth();
  APInt converted = *value;
  if (outWidth > converted.getBitWidth()) {
    converted = isSigned ? converted.sext(outWidth) : converted.zext(outWidth);
  } else if (outWidth < converted.getBitWidth()) {
    converted = converted.trunc(outWidth);
  }
  return bindIntegerLike(out, converted, classical);
}

[[nodiscard]] static bool isSupportedClassicalType(Type type) {
  return isa<IndexType, IntegerType, FloatType>(type);
}

/// Resolve a one-dimensional classical memref and a concrete index.
static FailureOr<ClassicalEnv::Scalar*>
lookupMemRefSlot(Value memref, Value index, ClassicalEnv& classical,
                 Operation* op) {
  const auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type || type.getRank() != 1 ||
      !isSupportedClassicalType(type.getElementType())) {
    return op->emitError()
           << "QCO DD simulation only supports one-dimensional memrefs of "
              "integer, index, or floating-point values";
  }
  auto idx = lookupIndex(index, classical, op);
  if (failed(idx)) {
    return failure();
  }
  auto it = classical.memrefs.find(memref);
  if (it == classical.memrefs.end() || !it->second->live) {
    return op->emitError()
           << "classical memref is not mapped for QCO DD simulation";
  }
  if (*idx < 0 || static_cast<size_t>(*idx) >= it->second->values.size()) {
    return op->emitError()
           << "classical memref index out of range for QCO DD simulation";
  }
  return &it->second->values[static_cast<size_t>(*idx)];
}

static LogicalResult applyMemRefAlloc(memref::AllocOp alloc,
                                      ClassicalEnv& classical) {
  const auto type = dyn_cast<MemRefType>(alloc.getType());
  if (!type || type.getRank() != 1 ||
      !isSupportedClassicalType(type.getElementType())) {
    return alloc.emitError()
           << "QCO DD simulation only supports one-dimensional memrefs of "
              "integer, index, or floating-point values";
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
    return alloc.emitError() << "classical memref size must be non-negative "
                                "for QCO DD simulation";
  }
  ClassicalEnv::Scalar zero;
  Type elementType = type.getElementType();
  if (elementType.isInteger(1)) {
    zero = false;
  } else if (isa<IndexType>(elementType)) {
    zero = int64_t{0};
  } else if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    zero = APInt(integerType.getWidth(), 0);
  } else {
    zero = 0.0;
  }
  classical.memrefs[alloc.getResult()] =
      std::make_shared<ClassicalEnv::MemRefStorage>(ClassicalEnv::MemRefStorage{
          .values = SmallVector<ClassicalEnv::Scalar>(static_cast<size_t>(size),
                                                      std::move(zero))});
  return success();
}

static LogicalResult applyMemRefStore(memref::StoreOp store,
                                      ClassicalEnv& classical) {
  if (store.getIndices().size() != 1) {
    return store.emitError()
           << "QCO DD simulation only supports 1-D memref.store";
  }
  auto slot = lookupMemRefSlot(store.getMemref(), store.getIndices()[0],
                               classical, store);
  if (failed(slot)) {
    return failure();
  }
  Value value = store.getValue();
  if (value.getType().isInteger(1)) {
    auto concrete = lookupBool(value, classical, store);
    if (failed(concrete)) {
      return failure();
    }
    **slot = *concrete;
  } else if (isa<IndexType>(value.getType())) {
    auto concrete = lookupIndex(value, classical, store);
    if (failed(concrete)) {
      return failure();
    }
    **slot = *concrete;
  } else if (isa<IntegerType>(value.getType())) {
    auto concrete = lookupInteger(value, classical, store);
    if (failed(concrete)) {
      return failure();
    }
    **slot = *concrete;
  } else if (isa<FloatType>(value.getType())) {
    auto concrete = lookupFloat(value, classical, store);
    if (failed(concrete)) {
      return failure();
    }
    **slot = *concrete;
  } else {
    return store.emitError() << "unsupported classical memref element type";
  }
  return success();
}

static LogicalResult applyMemRefLoad(memref::LoadOp load,
                                     ClassicalEnv& classical) {
  if (load.getIndices().size() != 1) {
    return load.emitError()
           << "QCO DD simulation only supports 1-D memref.load";
  }
  auto slot =
      lookupMemRefSlot(load.getMemref(), load.getIndices()[0], classical, load);
  if (failed(slot)) {
    return failure();
  }
  Type type = load.getType();
  if (type.isInteger(1)) {
    classical.bools[load.getResult()] = std::get<bool>(**slot);
  } else if (isa<IndexType>(type)) {
    classical.indices[load.getResult()] = std::get<int64_t>(**slot);
  } else if (isa<IntegerType>(type)) {
    classical.integers[load.getResult()] = std::get<APInt>(**slot);
  } else if (isa<FloatType>(type)) {
    classical.floats[load.getResult()] = std::get<double>(**slot);
  } else {
    return load.emitError() << "unsupported classical memref element type";
  }
  return success();
}

template <typename OpTy>
static LogicalResult applyBinaryI1(OpTy op, ClassicalEnv& classical,
                                   bool (*combine)(bool, bool)) {
  if (!op.getType().isInteger(1)) {
    return op.emitError() << "QCO DD simulation only supports i1 "
                          << op.getOperationName();
  }
  auto lhs = lookupBool(op.getLhs(), classical, op);
  auto rhs = lookupBool(op.getRhs(), classical, op);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  classical.bools[op.getResult()] = combine(*lhs, *rhs);
  return success();
}

template <typename OpTy>
static LogicalResult applyBinaryIndex(OpTy op, ClassicalEnv& classical,
                                      int64_t (*combine)(int64_t, int64_t)) {
  if (!isa<IndexType>(op.getType())) {
    return op.emitError() << "QCO DD simulation only supports index "
                          << op.getOperationName();
  }
  auto lhs = lookupIndex(op.getLhs(), classical, op);
  auto rhs = lookupIndex(op.getRhs(), classical, op);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  classical.indices[op.getResult()] = combine(*lhs, *rhs);
  return success();
}

template <typename OpTy, typename Combine>
static LogicalResult applyBinaryInteger(OpTy op, ClassicalEnv& classical,
                                        Combine combine) {
  if (!isa<IntegerType>(op.getType()) || op.getType().isInteger(1)) {
    return op.emitError() << "expected a concrete integer operation";
  }
  auto lhs = lookupInteger(op.getLhs(), classical, op);
  auto rhs = lookupInteger(op.getRhs(), classical, op);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  classical.integers[op.getResult()] = combine(*lhs, *rhs);
  return success();
}

template <typename OpTy, typename Combine>
static LogicalResult applyBinaryFloat(OpTy op, ClassicalEnv& classical,
                                      Combine combine) {
  auto lhs = lookupFloat(op.getLhs(), classical, op);
  auto rhs = lookupFloat(op.getRhs(), classical, op);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  classical.floats[op.getResult()] = combine(*lhs, *rhs);
  return success();
}

template <typename OpTy, typename Apply>
static LogicalResult applyUnaryFloat(OpTy op, ClassicalEnv& classical,
                                     Apply apply) {
  auto value = lookupFloat(op.getOperand(), classical, op);
  if (failed(value)) {
    return failure();
  }
  classical.floats[op.getResult()] = apply(*value);
  return success();
}

template <typename OpTy, typename Combine>
static LogicalResult applyBinaryIntegerLike(OpTy op, ClassicalEnv& classical,
                                            Combine combine) {
  auto lhs = lookupIntegerLike(op.getLhs(), classical, op);
  auto rhs = lookupIntegerLike(op.getRhs(), classical, op);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  if (rhs->isZero()) {
    return op.emitError() << "division by zero during QCO DD simulation";
  }
  return bindIntegerLike(op.getResult(), combine(*lhs, *rhs), classical);
}

static LogicalResult applyClassicalOp(Operation& op, ClassicalEnv& classical) {
  return TypeSwitch<Operation*, LogicalResult>(&op)
      .Case<arith::AndIOp>([&](arith::AndIOp andOp) -> LogicalResult {
        if (andOp.getType().isInteger(1)) {
          return applyBinaryI1(andOp, classical,
                               [](bool a, bool b) { return a && b; });
        }
        if (isa<IntegerType>(andOp.getType())) {
          return applyBinaryInteger(
              andOp, classical,
              [](const APInt& a, const APInt& b) { return a & b; });
        }
        return applyBinaryIndex(andOp, classical,
                                [](int64_t a, int64_t b) { return a & b; });
      })
      .Case<arith::OrIOp>([&](arith::OrIOp orOp) -> LogicalResult {
        if (orOp.getType().isInteger(1)) {
          return applyBinaryI1(orOp, classical,
                               [](bool a, bool b) { return a || b; });
        }
        if (isa<IntegerType>(orOp.getType())) {
          return applyBinaryInteger(
              orOp, classical,
              [](const APInt& a, const APInt& b) { return a | b; });
        }
        return applyBinaryIndex(orOp, classical,
                                [](int64_t a, int64_t b) { return a | b; });
      })
      .Case<arith::XOrIOp>([&](arith::XOrIOp xorOp) -> LogicalResult {
        if (xorOp.getType().isInteger(1)) {
          return applyBinaryI1(xorOp, classical,
                               [](bool a, bool b) { return a != b; });
        }
        if (isa<IntegerType>(xorOp.getType())) {
          return applyBinaryInteger(
              xorOp, classical,
              [](const APInt& a, const APInt& b) { return a ^ b; });
        }
        return applyBinaryIndex(xorOp, classical,
                                [](int64_t a, int64_t b) { return a ^ b; });
      })
      .Case<arith::AddIOp>([&](arith::AddIOp addOp) {
        if (addOp.getType().isInteger(1)) {
          return applyBinaryI1(addOp, classical,
                               [](bool a, bool b) { return a != b; });
        }
        if (isa<IntegerType>(addOp.getType())) {
          return applyBinaryInteger(
              addOp, classical,
              [](const APInt& a, const APInt& b) { return a + b; });
        }
        return applyBinaryIndex(addOp, classical, [](int64_t a, int64_t b) {
          return static_cast<int64_t>(static_cast<uint64_t>(a) +
                                      static_cast<uint64_t>(b));
        });
      })
      .Case<arith::SubIOp>([&](arith::SubIOp subOp) {
        if (subOp.getType().isInteger(1)) {
          return applyBinaryI1(subOp, classical,
                               [](bool a, bool b) { return a != b; });
        }
        if (isa<IntegerType>(subOp.getType())) {
          return applyBinaryInteger(
              subOp, classical,
              [](const APInt& a, const APInt& b) { return a - b; });
        }
        return applyBinaryIndex(subOp, classical, [](int64_t a, int64_t b) {
          return static_cast<int64_t>(static_cast<uint64_t>(a) -
                                      static_cast<uint64_t>(b));
        });
      })
      .Case<arith::MulIOp>([&](arith::MulIOp mulOp) {
        if (mulOp.getType().isInteger(1)) {
          return applyBinaryI1(mulOp, classical,
                               [](bool a, bool b) { return a && b; });
        }
        if (isa<IntegerType>(mulOp.getType())) {
          return applyBinaryInteger(
              mulOp, classical,
              [](const APInt& a, const APInt& b) { return a * b; });
        }
        return applyBinaryIndex(mulOp, classical, [](int64_t a, int64_t b) {
          return static_cast<int64_t>(static_cast<uint64_t>(a) *
                                      static_cast<uint64_t>(b));
        });
      })
      .Case<arith::DivUIOp>([&](arith::DivUIOp div) {
        return applyBinaryIntegerLike(
            div, classical,
            [](const APInt& a, const APInt& b) { return a.udiv(b); });
      })
      .Case<arith::DivSIOp>([&](arith::DivSIOp div) {
        return applyBinaryIntegerLike(
            div, classical,
            [](const APInt& a, const APInt& b) { return a.sdiv(b); });
      })
      .Case<arith::RemUIOp>([&](arith::RemUIOp rem) {
        return applyBinaryIntegerLike(
            rem, classical,
            [](const APInt& a, const APInt& b) { return a.urem(b); });
      })
      .Case<arith::RemSIOp>([&](arith::RemSIOp rem) {
        return applyBinaryIntegerLike(
            rem, classical,
            [](const APInt& a, const APInt& b) { return a.srem(b); });
      })
      .Case<arith::MaxSIOp>([&](arith::MaxSIOp maximum) {
        auto lhs = lookupIntegerLike(maximum.getLhs(), classical, maximum);
        auto rhs = lookupIntegerLike(maximum.getRhs(), classical, maximum);
        if (failed(lhs) || failed(rhs)) {
          return failure();
        }
        return bindIntegerLike(maximum.getResult(),
                               lhs->sgt(*rhs) ? *lhs : *rhs, classical);
      })
      .Case<arith::MinSIOp>([&](arith::MinSIOp minimum) {
        auto lhs = lookupIntegerLike(minimum.getLhs(), classical, minimum);
        auto rhs = lookupIntegerLike(minimum.getRhs(), classical, minimum);
        if (failed(lhs) || failed(rhs)) {
          return failure();
        }
        return bindIntegerLike(minimum.getResult(),
                               lhs->slt(*rhs) ? *lhs : *rhs, classical);
      })
      .Case<arith::MaxUIOp>([&](arith::MaxUIOp maximum) {
        auto lhs = lookupIntegerLike(maximum.getLhs(), classical, maximum);
        auto rhs = lookupIntegerLike(maximum.getRhs(), classical, maximum);
        if (failed(lhs) || failed(rhs)) {
          return failure();
        }
        return bindIntegerLike(maximum.getResult(),
                               lhs->ugt(*rhs) ? *lhs : *rhs, classical);
      })
      .Case<arith::MinUIOp>([&](arith::MinUIOp minimum) {
        auto lhs = lookupIntegerLike(minimum.getLhs(), classical, minimum);
        auto rhs = lookupIntegerLike(minimum.getRhs(), classical, minimum);
        if (failed(lhs) || failed(rhs)) {
          return failure();
        }
        return bindIntegerLike(minimum.getResult(),
                               lhs->ult(*rhs) ? *lhs : *rhs, classical);
      })
      .Case<arith::ShLIOp>([&](arith::ShLIOp shli) -> LogicalResult {
        if (auto integerType = dyn_cast<IntegerType>(shli.getType())) {
          auto lhs = lookupInteger(shli.getLhs(), classical, shli);
          auto rhs = lookupInteger(shli.getRhs(), classical, shli);
          if (failed(lhs) || failed(rhs)) {
            return failure();
          }
          if (rhs->uge(integerType.getWidth())) {
            return shli.emitError()
                   << "shift amount out of range for QCO DD simulation";
          }
          classical.integers[shli.getResult()] = lhs->shl(rhs->getZExtValue());
          return success();
        }
        if (!isa<IndexType>(shli.getType())) {
          return shli.emitError() << "QCO DD simulation only supports index "
                                  << arith::ShLIOp::getOperationName();
        }
        auto lhs = lookupIndex(shli.getLhs(), classical, shli);
        auto rhs = lookupIndex(shli.getRhs(), classical, shli);
        if (failed(lhs) || failed(rhs)) {
          return failure();
        }
        if (*rhs < 0 || *rhs >= 64) {
          return shli.emitError()
                 << "shift amount out of range for QCO DD simulation";
        }
        classical.indices[shli.getResult()] = static_cast<int64_t>(
            static_cast<uint64_t>(*lhs) << static_cast<unsigned>(*rhs));
        return success();
      })
      .Case<arith::ShRUIOp>([&](arith::ShRUIOp shrui) -> LogicalResult {
        if (auto integerType = dyn_cast<IntegerType>(shrui.getType())) {
          auto lhs = lookupInteger(shrui.getLhs(), classical, shrui);
          auto rhs = lookupInteger(shrui.getRhs(), classical, shrui);
          if (failed(lhs) || failed(rhs)) {
            return failure();
          }
          if (rhs->uge(integerType.getWidth())) {
            return shrui.emitError()
                   << "shift amount out of range for QCO DD simulation";
          }
          classical.integers[shrui.getResult()] =
              lhs->lshr(rhs->getZExtValue());
          return success();
        }
        if (!isa<IndexType>(shrui.getType())) {
          return shrui.emitError() << "QCO DD simulation only supports index "
                                   << arith::ShRUIOp::getOperationName();
        }
        auto lhs = lookupIndex(shrui.getLhs(), classical, shrui);
        auto rhs = lookupIndex(shrui.getRhs(), classical, shrui);
        if (failed(lhs) || failed(rhs)) {
          return failure();
        }
        if (*rhs < 0 || *rhs >= 64) {
          return shrui.emitError()
                 << "shift amount out of range for QCO DD simulation";
        }
        classical.indices[shrui.getResult()] = static_cast<int64_t>(
            static_cast<uint64_t>(*lhs) >> static_cast<unsigned>(*rhs));
        return success();
      })
      .Case<arith::ShRSIOp>([&](arith::ShRSIOp shrsi) -> LogicalResult {
        auto lhs = lookupIntegerLike(shrsi.getLhs(), classical, shrsi);
        auto rhs = lookupIntegerLike(shrsi.getRhs(), classical, shrsi);
        if (failed(lhs) || failed(rhs)) {
          return failure();
        }
        if (rhs->uge(lhs->getBitWidth())) {
          return shrsi.emitError()
                 << "shift amount out of range for QCO DD simulation";
        }
        return bindIntegerLike(shrsi.getResult(),
                               lhs->ashr(rhs->getZExtValue()), classical);
      })
      .Case<arith::CmpIOp>([&](arith::CmpIOp cmp) -> LogicalResult {
        if (!cmp.getType().isInteger(1)) {
          return cmp.emitError()
                 << "QCO DD simulation only supports cmpi to i1";
        }
        FailureOr<int64_t> lhs;
        FailureOr<int64_t> rhs;
        if (isa<IndexType>(cmp.getLhs().getType())) {
          lhs = lookupIndex(cmp.getLhs(), classical, cmp);
          rhs = lookupIndex(cmp.getRhs(), classical, cmp);
        } else if (cmp.getLhs().getType().isInteger(1)) {
          auto lb = lookupBool(cmp.getLhs(), classical, cmp);
          auto rb = lookupBool(cmp.getRhs(), classical, cmp);
          if (failed(lb) || failed(rb)) {
            return failure();
          }
          // Unsigned i1 uses 0/1; signed predicates use arith sign-extension
          // (true → -1). Equality is identical under either encoding.
          const int64_t aU = *lb ? 1 : 0;
          const int64_t bU = *rb ? 1 : 0;
          const int64_t aS = *lb ? -1 : 0;
          const int64_t bS = *rb ? -1 : 0;
          bool result = false;
          switch (cmp.getPredicate()) {
          case arith::CmpIPredicate::eq:
            result = aU == bU;
            break;
          case arith::CmpIPredicate::ne:
            result = aU != bU;
            break;
          case arith::CmpIPredicate::slt:
            result = aS < bS;
            break;
          case arith::CmpIPredicate::sle:
            result = aS <= bS;
            break;
          case arith::CmpIPredicate::sgt:
            result = aS > bS;
            break;
          case arith::CmpIPredicate::sge:
            result = aS >= bS;
            break;
          case arith::CmpIPredicate::ult:
            result = static_cast<uint64_t>(aU) < static_cast<uint64_t>(bU);
            break;
          case arith::CmpIPredicate::ule:
            result = static_cast<uint64_t>(aU) <= static_cast<uint64_t>(bU);
            break;
          case arith::CmpIPredicate::ugt:
            result = static_cast<uint64_t>(aU) > static_cast<uint64_t>(bU);
            break;
          case arith::CmpIPredicate::uge:
            result = static_cast<uint64_t>(aU) >= static_cast<uint64_t>(bU);
            break;
          }
          classical.bools[cmp.getResult()] = result;
          return success();
        } else if (isa<IntegerType>(cmp.getLhs().getType())) {
          auto a = lookupInteger(cmp.getLhs(), classical, cmp);
          auto b = lookupInteger(cmp.getRhs(), classical, cmp);
          if (failed(a) || failed(b)) {
            return failure();
          }
          bool result = false;
          switch (cmp.getPredicate()) {
          case arith::CmpIPredicate::eq:
            result = *a == *b;
            break;
          case arith::CmpIPredicate::ne:
            result = *a != *b;
            break;
          case arith::CmpIPredicate::slt:
            result = a->slt(*b);
            break;
          case arith::CmpIPredicate::sle:
            result = a->sle(*b);
            break;
          case arith::CmpIPredicate::sgt:
            result = a->sgt(*b);
            break;
          case arith::CmpIPredicate::sge:
            result = a->sge(*b);
            break;
          case arith::CmpIPredicate::ult:
            result = a->ult(*b);
            break;
          case arith::CmpIPredicate::ule:
            result = a->ule(*b);
            break;
          case arith::CmpIPredicate::ugt:
            result = a->ugt(*b);
            break;
          case arith::CmpIPredicate::uge:
            result = a->uge(*b);
            break;
          }
          classical.bools[cmp.getResult()] = result;
          return success();
        } else {
          return cmp.emitError()
                 << "QCO DD simulation only supports cmpi on integers or index";
        }
        if (failed(lhs) || failed(rhs)) {
          return failure();
        }
        const int64_t a = *lhs;
        const int64_t b = *rhs;
        bool result = false;
        switch (cmp.getPredicate()) {
        case arith::CmpIPredicate::eq:
          result = a == b;
          break;
        case arith::CmpIPredicate::ne:
          result = a != b;
          break;
        case arith::CmpIPredicate::slt:
          result = a < b;
          break;
        case arith::CmpIPredicate::sle:
          result = a <= b;
          break;
        case arith::CmpIPredicate::sgt:
          result = a > b;
          break;
        case arith::CmpIPredicate::sge:
          result = a >= b;
          break;
        case arith::CmpIPredicate::ult:
          result = static_cast<uint64_t>(a) < static_cast<uint64_t>(b);
          break;
        case arith::CmpIPredicate::ule:
          result = static_cast<uint64_t>(a) <= static_cast<uint64_t>(b);
          break;
        case arith::CmpIPredicate::ugt:
          result = static_cast<uint64_t>(a) > static_cast<uint64_t>(b);
          break;
        case arith::CmpIPredicate::uge:
          result = static_cast<uint64_t>(a) >= static_cast<uint64_t>(b);
          break;
        }
        classical.bools[cmp.getResult()] = result;
        return success();
      })
      .Case<arith::SelectOp>([&](arith::SelectOp select) -> LogicalResult {
        auto cond = lookupBool(select.getCondition(), classical, select);
        if (failed(cond)) {
          return failure();
        }
        if (select.getType().isInteger(1)) {
          auto t = lookupBool(select.getTrueValue(), classical, select);
          auto f = lookupBool(select.getFalseValue(), classical, select);
          if (failed(t) || failed(f)) {
            return failure();
          }
          classical.bools[select.getResult()] = *cond ? *t : *f;
          return success();
        }
        if (isa<IndexType>(select.getType())) {
          auto t = lookupIndex(select.getTrueValue(), classical, select);
          auto f = lookupIndex(select.getFalseValue(), classical, select);
          if (failed(t) || failed(f)) {
            return failure();
          }
          classical.indices[select.getResult()] = *cond ? *t : *f;
          return success();
        }
        if (isa<IntegerType>(select.getType())) {
          auto t = lookupInteger(select.getTrueValue(), classical, select);
          auto f = lookupInteger(select.getFalseValue(), classical, select);
          if (failed(t) || failed(f)) {
            return failure();
          }
          classical.integers[select.getResult()] = *cond ? *t : *f;
          return success();
        }
        if (isa<FloatType>(select.getType())) {
          auto t = lookupFloat(select.getTrueValue(), classical, select);
          auto f = lookupFloat(select.getFalseValue(), classical, select);
          if (failed(t) || failed(f)) {
            return failure();
          }
          classical.floats[select.getResult()] = *cond ? *t : *f;
          return success();
        }
        return select.emitError()
               << "QCO DD simulation only supports select on scalar values";
      })
      .Case<arith::ExtUIOp>([&](arith::ExtUIOp ext) {
        return applyIntegerCast(ext.getIn(), ext.getOut(), ext, classical,
                                /*isSigned=*/false);
      })
      .Case<arith::ExtSIOp>([&](arith::ExtSIOp ext) {
        return applyIntegerCast(ext.getIn(), ext.getOut(), ext, classical,
                                /*isSigned=*/true);
      })
      .Case<arith::IndexCastUIOp>([&](arith::IndexCastUIOp cast) {
        return applyIntegerCast(cast.getIn(), cast.getOut(), cast, classical,
                                /*isSigned=*/false);
      })
      .Case<arith::IndexCastOp>([&](arith::IndexCastOp cast) {
        return applyIntegerCast(cast.getIn(), cast.getOut(), cast, classical,
                                /*isSigned=*/true);
      })
      .Case<arith::TruncIOp>([&](arith::TruncIOp trunc) {
        return applyIntegerCast(trunc.getIn(), trunc.getOut(), trunc, classical,
                                /*isSigned=*/false);
      })
      .Case<arith::AddFOp>([&](arith::AddFOp add) {
        return applyBinaryFloat(add, classical,
                                [](double a, double b) { return a + b; });
      })
      .Case<arith::SubFOp>([&](arith::SubFOp sub) {
        return applyBinaryFloat(sub, classical,
                                [](double a, double b) { return a - b; });
      })
      .Case<arith::MulFOp>([&](arith::MulFOp mul) {
        return applyBinaryFloat(mul, classical,
                                [](double a, double b) { return a * b; });
      })
      .Case<arith::DivFOp>([&](arith::DivFOp div) {
        return applyBinaryFloat(div, classical,
                                [](double a, double b) { return a / b; });
      })
      .Case<arith::RemFOp>([&](arith::RemFOp rem) {
        return applyBinaryFloat(
            rem, classical, [](double a, double b) { return std::fmod(a, b); });
      })
      .Case<arith::MaximumFOp>([&](arith::MaximumFOp maximum) {
        return applyBinaryFloat(maximum, classical, [](double a, double b) {
          if (std::isnan(a) || std::isnan(b)) {
            return std::numeric_limits<double>::quiet_NaN();
          }
          return std::fmax(a, b);
        });
      })
      .Case<arith::MinimumFOp>([&](arith::MinimumFOp minimum) {
        return applyBinaryFloat(minimum, classical, [](double a, double b) {
          if (std::isnan(a) || std::isnan(b)) {
            return std::numeric_limits<double>::quiet_NaN();
          }
          return std::fmin(a, b);
        });
      })
      .Case<arith::MaxNumFOp>([&](arith::MaxNumFOp maximum) {
        return applyBinaryFloat(maximum, classical, [](double a, double b) {
          return std::fmax(a, b);
        });
      })
      .Case<arith::MinNumFOp>([&](arith::MinNumFOp minimum) {
        return applyBinaryFloat(minimum, classical, [](double a, double b) {
          return std::fmin(a, b);
        });
      })
      .Case<arith::NegFOp>([&](arith::NegFOp neg) -> LogicalResult {
        auto value = lookupFloat(neg.getOperand(), classical, neg);
        if (failed(value)) {
          return failure();
        }
        classical.floats[neg.getResult()] = -*value;
        return success();
      })
      .Case<arith::CmpFOp>([&](arith::CmpFOp cmp) -> LogicalResult {
        auto lhs = lookupFloat(cmp.getLhs(), classical, cmp);
        auto rhs = lookupFloat(cmp.getRhs(), classical, cmp);
        if (failed(lhs) || failed(rhs)) {
          return failure();
        }
        classical.bools[cmp.getResult()] = arith::applyCmpPredicate(
            cmp.getPredicate(), APFloat(*lhs), APFloat(*rhs));
        return success();
      })
      .Case<arith::SIToFPOp, arith::UIToFPOp>(
          [&](Operation* castOp) -> LogicalResult {
            Value in = castOp->getOperand(0);
            auto value = lookupIntegerLike(in, classical, castOp);
            if (failed(value)) {
              return failure();
            }
            const bool isSigned = isa<arith::SIToFPOp>(castOp);
            classical.floats[castOp->getResult(0)] =
                value->roundToDouble(isSigned);
            return success();
          })
      .Case<arith::FPToSIOp, arith::FPToUIOp>(
          [&](Operation* castOp) -> LogicalResult {
            auto value = lookupFloat(castOp->getOperand(0), classical, castOp);
            if (failed(value)) {
              return failure();
            }
            Value out = castOp->getResult(0);
            const unsigned width =
                isa<IndexType>(out.getType())
                    ? 64U
                    : cast<IntegerType>(out.getType()).getWidth();
            const bool isSigned = isa<arith::FPToSIOp>(castOp);
            APSInt result(width, /*isUnsigned=*/!isSigned);
            bool exact = false;
            const auto status = APFloat(*value).convertToInteger(
                result, APFloat::rmTowardZero, &exact);
            if ((status & APFloat::opInvalidOp) != 0) {
              return castOp->emitError()
                     << "floating-point value is outside the destination "
                        "integer range during QCO DD simulation";
            }
            return bindIntegerLike(out, result, classical);
          })
      .Case<math::AbsFOp>([&](math::AbsFOp abs) {
        return applyUnaryFloat(abs, classical,
                               [](double value) { return std::abs(value); });
      })
      .Case<math::CeilOp>([&](math::CeilOp ceil) {
        return applyUnaryFloat(ceil, classical,
                               [](double value) { return std::ceil(value); });
      })
      .Case<math::CosOp>([&](math::CosOp cos) {
        return applyUnaryFloat(cos, classical,
                               [](double value) { return std::cos(value); });
      })
      .Case<math::ExpOp>([&](math::ExpOp exp) {
        return applyUnaryFloat(exp, classical,
                               [](double value) { return std::exp(value); });
      })
      .Case<math::FloorOp>([&](math::FloorOp floor) {
        return applyUnaryFloat(floor, classical,
                               [](double value) { return std::floor(value); });
      })
      .Case<math::LogOp>([&](math::LogOp log) {
        return applyUnaryFloat(log, classical,
                               [](double value) { return std::log(value); });
      })
      .Case<math::SinOp>([&](math::SinOp sin) {
        return applyUnaryFloat(sin, classical,
                               [](double value) { return std::sin(value); });
      })
      .Case<math::SqrtOp>([&](math::SqrtOp sqrt) {
        return applyUnaryFloat(sqrt, classical,
                               [](double value) { return std::sqrt(value); });
      })
      .Case<math::TanOp>([&](math::TanOp tan) {
        return applyUnaryFloat(tan, classical,
                               [](double value) { return std::tan(value); });
      })
      .Case<math::PowFOp>([&](math::PowFOp pow) {
        return applyBinaryFloat(
            pow, classical, [](double a, double b) { return std::pow(a, b); });
      })
      .Default([](Operation* unsupported) {
        return unsupported->emitError()
               << "unsupported classical op for QCO DD simulation: "
               << unsupported->getName().getStringRef();
      });
}

static LogicalResult bindLinearArgs(ValueRange operands, Block& block,
                                    WalkState& walk, Operation* op) {
  if (operands.size() != block.getNumArguments()) {
    return op->emitError()
           << "region argument count does not match linear operands";
  }
  for (auto [operand, arg] : llvm::zip_equal(operands, block.getArguments())) {
    if (isa<QubitType>(arg.getType())) {
      const auto q = walk.qubits.lookup(operand);
      if (!q) {
        return op->emitError()
               << "qubit SSA value is not mapped for QCO DD construction";
      }
      walk.qubits.bind(arg, *q);
      continue;
    }
    if (isQTensorType(arg.getType())) {
      const auto* slots = walk.tensors.lookup(operand);
      if (slots == nullptr) {
        return op->emitError()
               << "qtensor SSA value is not mapped for QCO DD simulation";
      }
      walk.tensors.bind(arg, *slots);
      continue;
    }
    return op->emitError()
           << "unsupported linear region argument type for QCO DD simulation: "
           << arg.getType();
  }
  return success();
}

/// Bind each source SSA onto the corresponding destination via its concrete
/// qubit, qtensor, or classical environment. Callers must ensure equal sizes.
static LogicalResult bindValuePairs(ValueRange sources, ValueRange dests,
                                    WalkState& walk, Operation* op) {
  for (auto [src, dest] : llvm::zip_equal(sources, dests)) {
    if (isa<QubitType>(dest.getType())) {
      if (!isa<QubitType>(src.getType())) {
        return op->emitError()
               << "qubit/classical SSA type mismatch for QCO DD simulation";
      }
      const auto q = walk.qubits.lookup(src);
      if (!q) {
        return op->emitError()
               << "qubit SSA value is not mapped for QCO DD construction";
      }
      walk.qubits.bind(dest, *q);
    } else if (isQTensorType(dest.getType())) {
      if (!isQTensorType(src.getType())) {
        return op->emitError()
               << "qtensor/classical SSA type mismatch for QCO DD simulation";
      }
      const auto* slots = walk.tensors.lookup(src);
      if (slots == nullptr) {
        return op->emitError()
               << "qtensor SSA value is not mapped for QCO DD simulation";
      }
      walk.tensors.bind(dest, *slots);
    } else if (failed(walk.classical.bindFrom(src, dest, op))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult bindYieldResults(YieldOp yield,
                                      ValueRange classicalResults,
                                      ValueRange linearResults,
                                      WalkState& walk) {
  const size_t expected = classicalResults.size() + linearResults.size();
  if (yield.getNumOperands() != expected) {
    return yield.emitError()
           << "yield operand count does not match result segments";
  }
  size_t idx = 0;
  for (Value result : classicalResults) {
    if (failed(
            walk.classical.bindFrom(yield.getOperand(idx++), result, yield))) {
      return failure();
    }
  }
  for (Value result : linearResults) {
    Value yielded = yield.getOperand(idx++);
    if (isa<QubitType>(result.getType())) {
      const auto q = walk.qubits.lookup(yielded);
      if (!q) {
        return yield.emitError()
               << "yielded qubit SSA value is not mapped for QCO DD "
                  "construction";
      }
      walk.qubits.bind(result, *q);
      continue;
    }
    if (isQTensorType(result.getType())) {
      const auto* slots = walk.tensors.lookup(yielded);
      if (slots == nullptr) {
        return yield.emitError()
               << "yielded qtensor SSA value is not mapped for QCO DD "
                  "simulation";
      }
      walk.tensors.bind(result, *slots);
      continue;
    }
    return yield.emitError()
           << "unsupported linear result type for QCO DD simulation: "
           << result.getType();
  }
  return success();
}

template <typename StateDD>
static LogicalResult applyOp(Operation& op, WalkState& walk, StateDD& state);

template <typename StateDD>
static FailureOr<func::ReturnOp>
walkFunctionBody(func::FuncOp func, WalkState& walkState, StateDD& state);

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
  int64_t transitions = 0;
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
          lookupBool(branch.getCondition(), walk.classical, branch);
      if (failed(condition)) {
        return failure();
      }
      successor = *condition ? branch.getTrueDest() : branch.getFalseDest();
      successorOperands = *condition ? branch.getTrueDestOperands()
                                     : branch.getFalseDestOperands();
    } else if (auto switchOp = dyn_cast<cf::SwitchOp>(terminator)) {
      auto flag =
          lookupIntegerLike(switchOp.getFlag(), walk.classical, switchOp);
      if (failed(flag)) {
        return failure();
      }
      successor = switchOp.getDefaultDestination();
      successorOperands = switchOp.getDefaultOperands();
      if (const auto caseValues = switchOp.getCaseValues()) {
        for (auto [caseValue, destination, operands] : llvm::zip_equal(
                 caseValues->getValues<APInt>(), switchOp.getCaseDestinations(),
                 switchOp.getCaseOperands())) {
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
    if (transitions++ == MAX_CONTROL_FLOW_TRIPS) {
      return terminator->emitError()
             << scope << " CFG transition count exceeds QCO DD simulation "
             << "limit of " << MAX_CONTROL_FLOW_TRIPS;
    }
    block = successor;
  }
}

template <typename StateDD>
static LogicalResult
applyRegionBranch(ValueRange linearOperands, Block& block, YieldOp yield,
                  ValueRange classicalResults, ValueRange linearResults,
                  WalkState& walk, StateDD& state, Operation* parent) {
  if (failed(bindLinearArgs(linearOperands, block, walk, parent))) {
    return failure();
  }
  if (failed(walkBlock(block, walk, state))) {
    return failure();
  }
  return bindYieldResults(yield, classicalResults, linearResults, walk);
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
  if (!yield) {
    return parent->emitError() << "SCF region missing scf.yield";
  }
  if (yield.getNumOperands() != results.size()) {
    return parent->emitError()
           << "SCF yield operand count does not match operation results";
  }
  return bindValuePairs(yield.getOperands(), results, walk, parent);
}

template <typename StateDD>
static LogicalResult applyOp(Operation& op, WalkState& walk, StateDD& state) {
  return TypeSwitch<Operation*, LogicalResult>(&op)
      .template Case<StaticOp, SinkOp>([](auto) { return success(); })
      .template Case<arith::ConstantOp>([&](arith::ConstantOp constant) {
        return recordConstant(constant, walk.classical);
      })
      .template Case<AllocOp>([&](AllocOp alloc) -> LogicalResult {
        if constexpr (std::is_same_v<StateDD, dd::MatrixDD>) {
          return alloc.emitError()
                 << "dynamic qubit allocation is not supported for QCO DD "
                    "functionality construction";
        } else {
          auto slots = allocateZeroQubits(1, walk, state, alloc);
          if (failed(slots)) {
            return failure();
          }
          walk.qubits.bind(alloc.getResult(), *slots->front());
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
              auto size = lookupIndex(alloc.getSize(), walk.classical, alloc);
              if (failed(size)) {
                return failure();
              }
              if (*size <= 0) {
                return alloc.emitError()
                       << "qtensor allocation size must be positive for QCO "
                          "DD simulation";
              }
              auto slots = allocateZeroQubits(static_cast<size_t>(*size), walk,
                                              state, alloc);
              if (failed(slots)) {
                return failure();
              }
              walk.tensors.bind(alloc.getResult(), std::move(*slots));
              return success();
            }
          })
      .template Case<qtensor::FromElementsOp>(
          [&](qtensor::FromElementsOp fromElements) -> LogicalResult {
            auto wires = walk.qubits.lookupRange(fromElements.getElements(),
                                                 fromElements);
            if (failed(wires)) {
              return failure();
            }
            TensorSlots slots;
            slots.reserve(wires->size());
            for (qc::Qubit wire : *wires) {
              slots.emplace_back(wire);
            }
            walk.tensors.bind(fromElements.getResult(), std::move(slots));
            return success();
          })
      .template Case<qtensor::ExtractOp>([&](qtensor::ExtractOp extract)
                                             -> LogicalResult {
        const auto* inputSlots = walk.tensors.lookup(extract.getTensor());
        if (inputSlots == nullptr) {
          return extract.emitError()
                 << "qtensor SSA value is not mapped for QCO DD simulation";
        }
        auto index = lookupIndex(extract.getIndex(), walk.classical, extract);
        if (failed(index)) {
          return failure();
        }
        if (*index < 0 || static_cast<size_t>(*index) >= inputSlots->size()) {
          return extract.emitError()
                 << "qtensor index out of range for QCO DD simulation";
        }
        TensorSlots outputSlots = *inputSlots;
        auto& slot = outputSlots[static_cast<size_t>(*index)];
        if (!slot) {
          return extract.emitError()
                 << "qtensor element has already been extracted";
        }
        walk.qubits.bind(extract.getResult(), *slot);
        slot.reset();
        walk.tensors.bind(extract.getOutTensor(), std::move(outputSlots));
        return success();
      })
      .template Case<qtensor::InsertOp>(
          [&](qtensor::InsertOp insert) -> LogicalResult {
            const auto* inputSlots = walk.tensors.lookup(insert.getDest());
            if (inputSlots == nullptr) {
              return insert.emitError()
                     << "qtensor SSA value is not mapped for QCO DD simulation";
            }
            const auto wire = walk.qubits.lookup(insert.getScalar());
            if (!wire) {
              return insert.emitError()
                     << "inserted qubit SSA value is not mapped for QCO DD "
                        "simulation";
            }
            auto index = lookupIndex(insert.getIndex(), walk.classical, insert);
            if (failed(index)) {
              return failure();
            }
            if (*index < 0 ||
                static_cast<size_t>(*index) >= inputSlots->size()) {
              return insert.emitError()
                     << "qtensor index out of range for QCO DD simulation";
            }
            TensorSlots outputSlots = *inputSlots;
            outputSlots[static_cast<size_t>(*index)] = *wire;
            walk.tensors.bind(insert.getResult(), std::move(outputSlots));
            return success();
          })
      .template Case<qtensor::DeallocOp>(
          [&](qtensor::DeallocOp dealloc) -> LogicalResult {
            const auto* tracked = walk.tensors.lookup(dealloc.getTensor());
            if (tracked == nullptr) {
              return dealloc.emitError()
                     << "qtensor SSA value is not mapped for QCO DD simulation";
            }
            TensorSlots slots = *tracked;
            walk.tensors.erase(dealloc.getTensor());
            if constexpr (!std::is_same_v<StateDD, dd::MatrixDD>) {
              SmallVector<qc::Qubit> wires;
              for (const auto wire : slots) {
                if (wire) {
                  wires.push_back(*wire);
                }
              }
              llvm::sort(wires, [](qc::Qubit a, qc::Qubit b) { return a > b; });
              for (const qc::Qubit wire : wires) {
                if (failed(deallocateWire(wire, walk, state, dealloc))) {
                  return failure();
                }
              }
            }
            return success();
          })
      .template Case<memref::AllocOp>([&](memref::AllocOp alloc) {
        return applyMemRefAlloc(alloc, walk.classical);
      })
      .template Case<memref::StoreOp>([&](memref::StoreOp store) {
        return applyMemRefStore(store, walk.classical);
      })
      .template Case<memref::LoadOp>([&](memref::LoadOp load) {
        return applyMemRefLoad(load, walk.classical);
      })
      .template Case<memref::DeallocOp>(
          [&](memref::DeallocOp dealloc) -> LogicalResult {
            const auto it = walk.classical.memrefs.find(dealloc.getMemref());
            if (it == walk.classical.memrefs.end() || !it->second->live) {
              return dealloc.emitError()
                     << "classical memref is not mapped for QCO DD simulation";
            }
            it->second->live = false;
            return success();
          })
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
            return applyClassicalOp(*classicalOp, walk.classical);
          })
      .template Case<func::ReturnOp>([&](func::ReturnOp returnOp) {
        return validateReturn(returnOp, walk.qubits, walk.tensors);
      })
      .template Case<MeasureOp>([&](MeasureOp measureOp) -> LogicalResult {
        if constexpr (std::is_same_v<StateDD, dd::MatrixDD>) {
          return measureOp.emitError()
                 << "measurements are not supported for QCO DD functionality "
                    "construction";
        } else {
          if (walk.rng == nullptr) {
            return measureOp.emitError()
                   << "measurements require simulate(..., rng)";
          }
          const auto q = walk.qubits.lookup(measureOp.getQubitIn());
          if (!q) {
            return measureOp.emitError()
                   << "qubit SSA value is not mapped for QCO DD construction";
          }
          char bit;
          if constexpr (std::is_same_v<StateDD, dd::VectorDD>) {
            bit = walk.dd.measureOneCollapsing(state, *q, *walk.rng);
          } else {
            bit = measureDensity(state, *q, walk.qubits.numQubits, walk.dd,
                                 *walk.rng);
          }
          walk.classical.bools[measureOp.getResult()] = bit == '1';
          if (walk.classicalBits != nullptr) {
            walk.classicalBits->push_back(bit);
          }
          walk.qubits.bind(measureOp.getQubitOut(), *q);
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
          const auto q = walk.qubits.lookup(resetOp.getQubitIn());
          if (!q) {
            return resetOp.emitError()
                   << "qubit SSA value is not mapped for QCO DD construction";
          }
          char bit;
          if constexpr (std::is_same_v<StateDD, dd::VectorDD>) {
            bit = walk.dd.measureOneCollapsing(state, *q, *walk.rng);
          } else {
            bit = measureDensity(state, *q, walk.qubits.numQubits, walk.dd,
                                 *walk.rng);
          }
          if (bit == '1') {
            applyStateOperation(
                walk.dd.makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::X),
                                   *q),
                walk.dd, state);
          }
          walk.qubits.bind(resetOp.getQubitOut(), *q);
          return success();
        }
      })
      .template Case<IfOp>([&](IfOp ifOp) -> LogicalResult {
        const auto condIt = walk.classical.bools.find(ifOp.getCondition());
        if (condIt == walk.classical.bools.end()) {
          return ifOp.emitError()
                 << "if condition is not a concrete classical value";
        }
        Block* block = condIt->second ? ifOp.thenBlock() : ifOp.elseBlock();
        if (block == nullptr) {
          return ifOp.emitError() << "if region block is missing";
        }
        YieldOp yield = condIt->second ? ifOp.thenYield() : ifOp.elseYield();
        return applyRegionBranch(ifOp.getQubits(), *block, yield,
                                 ifOp.getClassicalResults(),
                                 ifOp.getLinearResults(), walk, state, ifOp);
      })
      .template Case<IndexSwitchOp>(
          [&](IndexSwitchOp switchOp) -> LogicalResult {
            const auto idxIt = walk.classical.indices.find(switchOp.getArg());
            if (idxIt == walk.classical.indices.end()) {
              return switchOp.emitError()
                     << "index_switch argument is not a concrete index";
            }
            const int64_t selector = idxIt->second;
            const auto cases = switchOp.getCases();
            if (switchOp.getDefaultRegion().empty()) {
              return switchOp.emitError()
                     << "index_switch default region is missing or empty";
            }
            Block* block = switchOp.getDefaultBlock();
            YieldOp yield = switchOp.getDefaultYield();
            if (block == nullptr) {
              return switchOp.emitError()
                     << "index_switch default region is missing or empty";
            }
            for (auto [i, caseValue] : llvm::enumerate(cases)) {
              if (caseValue == selector) {
                block = switchOp.getCaseBlock(i);
                yield = switchOp.getCaseYield(i);
                break;
              }
            }
            return applyRegionBranch(switchOp.getTargets(), *block, yield,
                                     switchOp.getClassicalResults(),
                                     switchOp.getLinearResults(), walk, state,
                                     switchOp);
          })
      .template Case<scf::IfOp>([&](scf::IfOp ifOp) -> LogicalResult {
        auto condition = lookupBool(ifOp.getCondition(), walk.classical, ifOp);
        if (failed(condition)) {
          return failure();
        }
        Region& selected =
            *condition ? ifOp.getThenRegion() : ifOp.getElseRegion();
        if (selected.empty()) {
          if (ifOp.getNumResults() != 0) {
            return ifOp.emitError()
                   << "selected empty scf.if region cannot produce results";
          }
          return success();
        }
        return applyScfRegion(selected, ifOp.getResults(), walk, state, ifOp);
      })
      .template Case<scf::IndexSwitchOp>(
          [&](scf::IndexSwitchOp switchOp) -> LogicalResult {
            auto selector =
                lookupIndex(switchOp.getArg(), walk.classical, switchOp);
            if (failed(selector)) {
              return failure();
            }
            Region* selected = &switchOp.getDefaultRegion();
            for (auto [i, caseValue] : llvm::enumerate(switchOp.getCases())) {
              if (caseValue == *selector) {
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
        auto lb = lookupIndex(forOp.getLowerBound(), walk.classical, forOp);
        auto ub = lookupIndex(forOp.getUpperBound(), walk.classical, forOp);
        auto step = lookupIndex(forOp.getStep(), walk.classical, forOp);
        if (failed(lb) || failed(ub) || failed(step)) {
          return failure();
        }
        if (*step <= 0) {
          return forOp.emitError()
                 << "scf.for step must be positive for QCO DD simulation";
        }
        int64_t trips = 0;
        if (*ub > *lb) {
          // Use unsigned arithmetic to avoid signed-overflow UB when
          // classical bounds are extreme (e.g. INT64_MIN / INT64_MAX).
          const auto span =
              static_cast<uint64_t>(*ub) - static_cast<uint64_t>(*lb);
          const uint64_t tripsU =
              ((span - 1) / static_cast<uint64_t>(*step)) + 1;
          if (tripsU > static_cast<uint64_t>(MAX_CONTROL_FLOW_TRIPS)) {
            return forOp.emitError()
                   << "scf.for trip count exceeds QCO DD simulation limit of "
                   << MAX_CONTROL_FLOW_TRIPS;
          }
          trips = static_cast<int64_t>(tripsU);
        }

        Block& body = *forOp.getBody();
        SmallVector<Value> carried(forOp.getInits().begin(),
                                   forOp.getInits().end());

        if (trips == 0) {
          if (carried.size() != forOp.getNumResults()) {
            return forOp.emitError()
                   << "scf.for result size mismatch during simulation";
          }
          return bindValuePairs(carried, forOp.getResults(), walk, forOp);
        }

        for (int64_t t = 0; t < trips; ++t) {
          const auto offset =
              static_cast<uint64_t>(t) * static_cast<uint64_t>(*step);
          walk.classical.indices[body.getArgument(0)] =
              static_cast<int64_t>(static_cast<uint64_t>(*lb) + offset);
          auto iterArgs = body.getArguments().drop_front();
          if (carried.size() != iterArgs.size()) {
            return forOp.emitError()
                   << "scf.for iter_args size mismatch during simulation";
          }
          if (failed(bindValuePairs(carried, iterArgs, walk, forOp))) {
            return failure();
          }
          if (failed(walkBlock(body, walk, state))) {
            return failure();
          }
          auto yield = dyn_cast<scf::YieldOp>(body.getTerminator());
          if (!yield) {
            return forOp.emitError() << "scf.for body missing scf.yield";
          }
          carried.assign(yield.getOperands().begin(),
                         yield.getOperands().end());
        }
        if (carried.size() != forOp.getNumResults()) {
          return forOp.emitError()
                 << "scf.for result size mismatch during simulation";
        }
        return bindValuePairs(carried, forOp.getResults(), walk, forOp);
      })
      .template Case<scf::WhileOp>([&](scf::WhileOp whileOp) -> LogicalResult {
        if (!whileOp.getBefore().hasOneBlock() ||
            !whileOp.getAfter().hasOneBlock()) {
          return whileOp.emitError()
                 << "scf.while regions must contain exactly one block for "
                    "QCO DD simulation";
        }

        Block& before = whileOp.getBefore().front();
        Block& after = whileOp.getAfter().front();
        SmallVector<Value> carried(whileOp.getInits().begin(),
                                   whileOp.getInits().end());
        int64_t trips = 0;

        while (true) {
          if (carried.size() != before.getNumArguments()) {
            return whileOp.emitError()
                   << "scf.while before-region argument size mismatch "
                      "during simulation";
          }
          if (failed(bindValuePairs(carried, before.getArguments(), walk,
                                    whileOp))) {
            return failure();
          }
          if (failed(walkBlock(before, walk, state))) {
            return failure();
          }
          auto condition = dyn_cast<scf::ConditionOp>(before.getTerminator());
          if (!condition) {
            return whileOp.emitError()
                   << "scf.while before region missing scf.condition";
          }
          auto conditionValue =
              lookupBool(condition.getCondition(), walk.classical, whileOp);
          if (failed(conditionValue)) {
            return failure();
          }
          if (!*conditionValue) {
            if (condition.getArgs().size() != whileOp.getNumResults()) {
              return whileOp.emitError()
                     << "scf.while result size mismatch during simulation";
            }
            return bindValuePairs(condition.getArgs(), whileOp.getResults(),
                                  walk, whileOp);
          }
          if (trips == MAX_CONTROL_FLOW_TRIPS) {
            return whileOp.emitError()
                   << "scf.while trip count exceeds QCO DD simulation limit of "
                   << MAX_CONTROL_FLOW_TRIPS;
          }
          if (condition.getArgs().size() != after.getNumArguments()) {
            return whileOp.emitError()
                   << "scf.while after-region argument size mismatch during "
                      "simulation";
          }
          if (failed(bindValuePairs(condition.getArgs(), after.getArguments(),
                                    walk, whileOp))) {
            return failure();
          }
          if (failed(walkBlock(after, walk, state))) {
            return failure();
          }
          auto yield = dyn_cast<scf::YieldOp>(after.getTerminator());
          if (!yield) {
            return whileOp.emitError()
                   << "scf.while after region missing scf.yield";
          }
          carried.assign(yield.getOperands().begin(),
                         yield.getOperands().end());
          ++trips;
        }
      })
      .template Case<func::CallOp>([&](func::CallOp call) -> LogicalResult {
        auto module = call->getParentOfType<ModuleOp>();
        if (!module) {
          return call.emitError()
                 << "func.call requires a parent ModuleOp for QCO DD "
                    "simulation";
        }
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (!callee) {
          return call.emitError()
                 << "func.call callee not found: " << call.getCallee();
        }
        if (callee.isDeclaration()) {
          return call.emitError() << "func.call callee must have a body";
        }
        if (walk.activeCalls == nullptr) {
          return call.emitError()
                 << "internal error: missing active call set for QCO DD";
        }
        Operation* calleeOp = callee.getOperation();
        if (!walk.activeCalls->insert(calleeOp).second) {
          return call.emitError()
                 << "recursive func.call is not supported for QCO DD "
                    "simulation";
        }
        ActiveCallGuard guard(walk.activeCalls, calleeOp);

        if (call.getArgOperands().size() != callee.getNumArguments()) {
          return call.emitError()
                 << "func.call operand count does not match callee arguments";
        }
        if (failed(bindValuePairs(call.getArgOperands(), callee.getArguments(),
                                  walk, call))) {
          return failure();
        }

        auto returnOp = walkFunctionBody(callee, walk, state);
        if (failed(returnOp)) {
          return failure();
        }

        // Map callee return operands onto call results via the return op.
        if (returnOp->getNumOperands() != call.getNumResults()) {
          return call.emitError()
                 << "func.call result count does not match callee return";
        }
        return bindValuePairs(returnOp->getOperands(), call.getResults(), walk,
                              call);
      })
      .template Case<CtrlOp>([&](CtrlOp ctrlOp) -> LogicalResult {
        if (auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(
                *ctrlOp.getBody())) {
          auto decoded = decodeStandardGate(inner, walk.classical);
          if (failed(decoded)) {
            return failure();
          }
          if (*decoded) {
            auto controlQubits =
                walk.qubits.lookupRange(ctrlOp.getControlsIn(), ctrlOp);
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
            auto decoded = decodeStandardGate(unitary, walk.classical);
            if (failed(decoded)) {
              return failure();
            }
            if (*decoded) {
              return applyDecodedStandard(unitary, **decoded, {}, walk, state);
            }
            return applyUnitaryMatrix(unitary, walk, state);
          })
      .Default([](Operation* unsupported) {
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
  return validateReturn(*returnOp, walk.qubits, walk.tensors);
}

struct PreparedState {
  QubitMap qubits;
  TensorMap tensors;
};

static FailureOr<PreparedState>
prepare(func::FuncOp func, const dd::Package& dd, const DDBindings& bindings) {
  if (func.isDeclaration()) {
    return func.emitError() << "QCO DD construction requires a function body";
  }
  PreparedState prepared;
  QubitMap& qubits = prepared.qubits;
  for (Block& block : func.getBody()) {
    for (StaticOp staticOp : block.getOps<StaticOp>()) {
      const auto q = static_cast<qc::Qubit>(staticOp.getIndex());
      qubits.bind(staticOp.getQubit(), q);
      qubits.numQubits = std::max(qubits.numQubits, static_cast<size_t>(q) + 1);
    }
  }
  // No `qco.static`: treat qubit-typed block arguments as wires `0..n-1`.
  if (qubits.numQubits == 0) {
    qc::Qubit next = 0;
    for (Value arg : func.getArguments()) {
      if (isa<QubitType>(arg.getType())) {
        qubits.bind(arg, next++);
      } else if (isQTensorType(arg.getType())) {
        const auto tensorType = cast<RankedTensorType>(arg.getType());
        int64_t size = tensorType.getDimSize(0);
        if (tensorType.isDynamicDim(0)) {
          const auto binding = bindings.find(arg);
          if (binding == bindings.end() || !isa<IntegerAttr>(binding->second)) {
            return func.emitError()
                   << "dynamic qtensor function arguments require an integer "
                      "extent in the QCO DD bindings";
          }
          size = cast<IntegerAttr>(binding->second).getInt();
          if (size < 0) {
            return func.emitError()
                   << "dynamic qtensor function argument extent must be "
                      "non-negative";
          }
        }
        TensorSlots slots;
        slots.reserve(static_cast<size_t>(size));
        for (int64_t i = 0; i < size; ++i) {
          slots.emplace_back(next++);
        }
        prepared.tensors.bind(arg, std::move(slots));
      }
    }
    qubits.numQubits = static_cast<size_t>(next);
  }
  if (dd.qubits() < qubits.numQubits) {
    return func.emitError() << "DD package has " << dd.qubits()
                            << " qubits but function uses " << qubits.numQubits;
  }
  return prepared;
}

FailureOr<dd::MatrixDD> buildFunctionality(func::FuncOp func, dd::Package& dd,
                                           const DDBindings& bindings) {
  auto preparedOr = prepare(func, dd, bindings);
  if (failed(preparedOr)) {
    return failure();
  }
  QubitMap qubits = std::move(preparedOr->qubits);
  TensorMap tensors = std::move(preparedOr->tensors);
  ClassicalEnv classical;
  if (failed(applyBindings(func, bindings, classical))) {
    return failure();
  }
  DenseSet<Operation*> activeCalls;
  WalkState walkState{.qubits = qubits,
                      .tensors = tensors,
                      .classical = classical,
                      .dd = dd,
                      .rng = nullptr,
                      .classicalBits = nullptr,
                      .activeCalls = &activeCalls};

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
             std::mt19937_64* rng, std::string* classicalBits,
             const DDBindings& bindings) {
  auto preparedOr = prepare(func, dd, bindings);
  if (failed(preparedOr)) {
    dd.decRef(in);
    return failure();
  }
  QubitMap qubits = std::move(preparedOr->qubits);
  TensorMap tensors = std::move(preparedOr->tensors);
  ClassicalEnv classical;
  if (failed(applyBindings(func, bindings, classical))) {
    dd.decRef(in);
    return failure();
  }
  DenseSet<Operation*> activeCalls;
  WalkState walkState{.qubits = qubits,
                      .tensors = tensors,
                      .classical = classical,
                      .dd = dd,
                      .rng = rng,
                      .classicalBits = classicalBits,
                      .activeCalls = &activeCalls};

  dd::VectorDD state = in;
  if (failed(walkFunction(func, walkState, state))) {
    dd.decRef(state);
    return failure();
  }
  return state;
}

FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd, const DDBindings& bindings) {
  return simulateImpl(func, in, dd, nullptr, nullptr, bindings);
}

FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd, std::mt19937_64& rng,
                                 const DDBindings& bindings) {
  return simulateImpl(func, in, dd, &rng, nullptr, bindings);
}

dd::MatrixDD makeDensityMatrix(const dd::VectorDD& state,
                               const size_t numQubits, dd::Package& dd) {
  std::map<std::tuple<dd::vNode*, dd::vNode*, int64_t>, dd::mCachedEdge> cache;
  const auto root = buildDensityMatrix(
      state, state, static_cast<int64_t>(numQubits) - 1, dd, cache);
  const auto density = dd::MatrixDD{.p = root.p, .w = dd.cn.lookup(root.w)};
  dd.incRef(density);
  return density;
}

struct DensitySimulationResult {
  dd::MatrixDD matrix;
  size_t numQubits;
};

static FailureOr<DensitySimulationResult>
simulateDensityImpl(func::FuncOp func, const dd::MatrixDD& in, dd::Package& dd,
                    std::mt19937_64* rng, std::string* classicalBits,
                    const DDBindings& bindings) {
  auto preparedOr = prepare(func, dd, bindings);
  if (failed(preparedOr)) {
    dd.decRef(in);
    return failure();
  }
  QubitMap qubits = std::move(preparedOr->qubits);
  TensorMap tensors = std::move(preparedOr->tensors);
  ClassicalEnv classical;
  if (failed(applyBindings(func, bindings, classical))) {
    dd.decRef(in);
    return failure();
  }
  DenseSet<Operation*> activeCalls;
  WalkState walkState{.qubits = qubits,
                      .tensors = tensors,
                      .classical = classical,
                      .dd = dd,
                      .rng = rng,
                      .classicalBits = classicalBits,
                      .activeCalls = &activeCalls};
  DensityState state{in};
  if (failed(walkFunction(func, walkState, state))) {
    dd.decRef(state.matrix);
    return failure();
  }
  return DensitySimulationResult{.matrix = state.matrix,
                                 .numQubits = walkState.qubits.numQubits};
}

FailureOr<dd::MatrixDD> simulateDensity(func::FuncOp func,
                                        const dd::MatrixDD& in, dd::Package& dd,
                                        const DDBindings& bindings) {
  auto result = simulateDensityImpl(func, in, dd, nullptr, nullptr, bindings);
  if (failed(result)) {
    return failure();
  }
  return result->matrix;
}

FailureOr<dd::MatrixDD> simulateDensity(func::FuncOp func,
                                        const dd::MatrixDD& in, dd::Package& dd,
                                        std::mt19937_64& rng,
                                        const DDBindings& bindings) {
  auto result = simulateDensityImpl(func, in, dd, &rng, nullptr, bindings);
  if (failed(result)) {
    return failure();
  }
  return result->matrix;
}

[[nodiscard]] static bool
requiresDynamicSampling(func::FuncOp func,
                        DenseSet<Operation*>* visiting = nullptr) {
  DenseSet<Operation*> localVisiting;
  DenseSet<Operation*>& active =
      visiting != nullptr ? *visiting : localVisiting;
  Operation* funcOp = func.getOperation();
  if (!active.insert(funcOp).second) {
    // Recursive call cycle: treat as dynamic to avoid infinite recursion.
    return true;
  }

  bool dynamic = false;
  func.walk([&](Operation* op) {
    // Only stochastic collapse forces per-shot re-simulation. Deterministic
    // control-flow can reuse a single simulated state.
    if (isa<MeasureOp, ResetOp>(op)) {
      dynamic = true;
      return WalkResult::interrupt();
    }
    if (auto call = dyn_cast<func::CallOp>(op)) {
      auto module = call->getParentOfType<ModuleOp>();
      if (!module) {
        dynamic = true;
        return WalkResult::interrupt();
      }
      auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
      if (!callee || requiresDynamicSampling(callee, &active)) {
        dynamic = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  active.erase(funcOp);
  return dynamic;
}

static FailureOr<SampleResult>
sampleImpl(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd,
           const size_t shots, std::mt19937_64& rng, const bool recordClassics,
           const DDBindings& bindings) {
  SampleResult result;
  if (shots == 0) {
    dd.decRef(in);
    return result;
  }

  if (!requiresDynamicSampling(func)) {
    auto stateOr = simulateImpl(func, in, dd, nullptr, nullptr, bindings);
    if (failed(stateOr)) {
      return failure();
    }
    dd::VectorDD state = *stateOr;
    for (size_t i = 0; i < shots; ++i) {
      result.shots[dd.measureAll(state, false, rng)] += 1;
    }
    dd.decRef(state);
    return result;
  }

  for (size_t i = 0; i < shots; ++i) {
    dd.incRef(in);
    std::string classical;
    auto stateOr = simulateImpl(
        func, in, dd, &rng, recordClassics ? &classical : nullptr, bindings);
    if (failed(stateOr)) {
      dd.decRef(in);
      return failure();
    }
    dd::VectorDD state = *stateOr;
    result.shots[dd.measureAll(state, false, rng)] += 1;
    if (recordClassics && !classical.empty()) {
      result.classical[classical] += 1;
    }
    dd.decRef(state);
  }
  dd.decRef(in);
  return result;
}

FailureOr<std::map<std::string, size_t>>
sampleDensity(func::FuncOp func, const dd::MatrixDD& in, dd::Package& dd,
              const size_t shots, std::mt19937_64& rng,
              const DDBindings& bindings) {
  std::map<std::string, size_t> result;
  if (shots == 0) {
    dd.decRef(in);
    return result;
  }

  if (!requiresDynamicSampling(func)) {
    auto simulated =
        simulateDensityImpl(func, in, dd, nullptr, nullptr, bindings);
    if (failed(simulated)) {
      return failure();
    }
    for (size_t i = 0; i < shots; ++i) {
      dd.incRef(simulated->matrix);
      DensityState sampleState{simulated->matrix};
      result[measureAllDensity(sampleState, simulated->numQubits, dd, rng)] +=
          1;
      dd.decRef(sampleState.matrix);
    }
    dd.decRef(simulated->matrix);
    return result;
  }

  for (size_t i = 0; i < shots; ++i) {
    dd.incRef(in);
    auto simulated = simulateDensityImpl(func, in, dd, &rng, nullptr, bindings);
    if (failed(simulated)) {
      dd.decRef(in);
      return failure();
    }
    DensityState sampleState{simulated->matrix};
    result[measureAllDensity(sampleState, simulated->numQubits, dd, rng)] += 1;
    dd.decRef(sampleState.matrix);
  }
  dd.decRef(in);
  return result;
}

FailureOr<std::map<std::string, size_t>>
sample(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd,
       const size_t shots, std::mt19937_64& rng, const DDBindings& bindings) {
  auto result =
      sampleImpl(func, in, dd, shots, rng, /*recordClassics=*/false, bindings);
  if (failed(result)) {
    return failure();
  }
  return std::move(result->shots);
}

FailureOr<SampleResult> sampleWithClassics(func::FuncOp func,
                                           const dd::VectorDD& in,
                                           dd::Package& dd, const size_t shots,
                                           std::mt19937_64& rng,
                                           const DDBindings& bindings) {
  return sampleImpl(func, in, dd, shots, rng, /*recordClassics=*/true,
                    bindings);
}

FailureOr<std::map<std::string, size_t>>
sample(func::FuncOp func, dd::Package& dd, const size_t shots,
       std::mt19937_64& rng, const DDBindings& bindings) {
  auto preparedOr = prepare(func, dd, bindings);
  if (failed(preparedOr)) {
    return failure();
  }
  const size_t n = preparedOr->qubits.numQubits;
  return sample(func, dd::makeZeroState(n, dd), dd, shots, rng, bindings);
}

FailureOr<SampleResult> sampleWithClassics(func::FuncOp func, dd::Package& dd,
                                           const size_t shots,
                                           std::mt19937_64& rng,
                                           const DDBindings& bindings) {
  auto preparedOr = prepare(func, dd, bindings);
  if (failed(preparedOr)) {
    return failure();
  }
  const size_t n = preparedOr->qubits.numQubits;
  return sampleWithClassics(func, dd::makeZeroState(n, dd), dd, shots, rng,
                            bindings);
}

} // namespace mlir::qco
