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

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
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
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlir::qco {
namespace {

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

struct ClassicalEnv {
  using RegisterState = std::vector<std::optional<bool>>;

  DenseMap<Value, llvm::APInt> scalars;
  /// Shared storage preserves CBit register identity across `func.call`.
  DenseMap<Value, std::shared_ptr<RegisterState>> registers;

  LogicalResult bindFrom(Value source, Value dest, Operation* op);
};

struct DecodedGate {
  qc::OpType type = qc::OpType::None;
  std::vector<dd::fp> params;
};

struct WalkState {
  QubitMap* qubits;
  ClassicalEnv* classical;
  dd::Package* dd;
  std::mt19937_64* rng = nullptr;
  bool deferTerminalMeasurements = false;
  std::string* classicalBits = nullptr;
  DenseSet<Operation*> activeCalls;
};

} // namespace

/// `std::nullopt` if @p unitary is not a standard gate; failure if its unitary
/// matrix is not known at compile time.
static FailureOr<std::optional<DecodedGate>>
decodeStandardGate(UnitaryOpInterface unitary) {
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
  if (!unitary.hasCompileTimeKnownUnitaryMatrix()) {
    return unitary.emitError()
           << "unitary must have a compile-time constant matrix";
  }

  DecodedGate decoded{.type = type, .params = {}};
  for (Value param : unitary.getParameters()) {
    decoded.params.push_back(
        static_cast<dd::fp>(*mlir::mqt::valueToDouble(param)));
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

template <typename StateDD>
static LogicalResult applyUnitaryMatrix(UnitaryOpInterface unitary,
                                        WalkState& walk, StateDD& state) {
  Operation* op = unitary.getOperation();
  if (!unitary.hasCompileTimeKnownUnitaryMatrix()) {
    return unitary.emitError()
           << "unitary must have a compile-time constant matrix";
  }
  if (auto gphase = dyn_cast<GPhaseOp>(op)) {
    const auto theta = *mlir::mqt::valueToDouble(gphase.getTheta());
    auto id = dd::Package::makeIdent();
    id.w = walk.dd->cn.lookup(std::cos(theta), std::sin(theta));
    state = walk.dd->applyOperation(id, state);
    return success();
  }
  if (isa<BarrierOp>(op)) {
    return walk.qubits->remapUnitary(unitary);
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
  const ArrayRef<qc::Qubit> wires = *wiresOr;
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
                                    const QubitMap& qubits) {
  qc::Qubit expected = 0;
  for (Value value : returnOp.getOperands()) {
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
  auto attr = dyn_cast<IntegerAttr>(constant.getValue());
  if (!attr) {
    return success();
  }
  if (constant.getType().isInteger(1)) {
    classical.scalars[constant.getResult()] =
        llvm::APInt(1, attr.getValue().isZero() ? 0 : 1);
  } else if (isa<IndexType>(constant.getType())) {
    classical.scalars[constant.getResult()] =
        llvm::APInt(64, static_cast<uint64_t>(attr.getInt()));
  }
  return success();
}

static FailureOr<llvm::APInt>
lookupInteger(Value value, ClassicalEnv& classical, Operation* op) {
  const auto it = classical.scalars.find(value);
  if (it != classical.scalars.end()) {
    return it->second;
  }
  if (value.getType().isInteger(1)) {
    return op->emitError()
           << "classical i1 SSA value is not mapped for QCO DD simulation";
  }
  if (isa<IndexType>(value.getType())) {
    return op->emitError()
           << "classical index SSA value is not mapped for QCO DD simulation";
  }
  return op->emitError()
         << "QCO DD simulation only supports integer operations on i1 or "
            "index";
}

static FailureOr<bool> lookupBool(Value value, ClassicalEnv& classical,
                                  Operation* op) {
  if (!value.getType().isInteger(1)) {
    return op->emitError()
           << "classical i1 SSA value is not mapped for QCO DD simulation";
  }
  auto result = lookupInteger(value, classical, op);
  if (failed(result)) {
    return failure();
  }
  return !result->isZero();
}

static FailureOr<int64_t> lookupIndex(Value value, ClassicalEnv& classical,
                                      Operation* op) {
  if (!isa<IndexType>(value.getType())) {
    return op->emitError()
           << "classical index SSA value is not mapped for QCO DD simulation";
  }
  auto result = lookupInteger(value, classical, op);
  if (failed(result)) {
    return failure();
  }
  return result->getSExtValue();
}

LogicalResult ClassicalEnv::bindFrom(Value source, Value dest, Operation* op) {
  if (dest.getType().isInteger(1)) {
    auto value = lookupBool(source, *this, op);
    if (failed(value)) {
      return failure();
    }
    scalars[dest] = llvm::APInt(1, *value ? 1 : 0);
    return success();
  }
  if (isa<IndexType>(dest.getType())) {
    auto value = lookupIndex(source, *this, op);
    if (failed(value)) {
      return failure();
    }
    scalars[dest] = llvm::APInt(64, static_cast<uint64_t>(*value));
    return success();
  }
  return op->emitError() << "unsupported classical type for QCO DD simulation: "
                         << dest.getType();
}

/// Cast a concrete `i1` SSA value to `index` via `index_castui`.
static LogicalResult applyI1ToIndex(Value in, Value out, Operation* op,
                                    ClassicalEnv& classical) {
  if (!isa<IndexType>(out.getType())) {
    return op->emitError()
           << "QCO DD simulation only supports casting i1 to index";
  }
  if (!in.getType().isInteger(1)) {
    return op->emitError()
           << "QCO DD simulation only supports casting from i1 to index";
  }
  auto bit = lookupBool(in, classical, op);
  if (failed(bit)) {
    return failure();
  }
  classical.scalars[out] = llvm::APInt(64, *bit ? 1 : 0);
  return success();
}

static LogicalResult allocateRegister(cbit::AllocOp alloc,
                                      ClassicalEnv& classical) {
  const auto width =
      static_cast<size_t>(alloc.getResult().getType().getWidth());
  const std::optional<bool> initialValue =
      alloc.getInitialization() == cbit::Initialization::Zero
          ? std::optional<bool>{false}
          : std::nullopt;
  classical.registers[alloc.getResult()] =
      std::make_shared<ClassicalEnv::RegisterState>(width, initialValue);
  return success();
}

static FailureOr<size_t> resolveRegisterIndex(Value index,
                                              cbit::RegisterType type,
                                              ClassicalEnv& classical,
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
  auto value = lookupBool(store.getValue(), classical, store);
  if (failed(index) || failed(value)) {
    return failure();
  }
  (*regIt->second)[*index] = *value;
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
  const auto value = (*regIt->second)[*index];
  if (!value) {
    return load.emitError() << "read from an undefined CBit register element";
  }
  classical.scalars[load.getResult()] = llvm::APInt(1, *value ? 1 : 0);
  return success();
}

template <typename OpTy, typename Combine>
static LogicalResult applyBinaryInteger(OpTy op, ClassicalEnv& classical,
                                        Combine combine,
                                        const bool indexOnly = false) {
  if (indexOnly && !isa<IndexType>(op.getType())) {
    return op.emitError() << "QCO DD simulation only supports index "
                          << op.getOperationName();
  }
  auto lhs = lookupInteger(op.getLhs(), classical, op);
  auto rhs = lookupInteger(op.getRhs(), classical, op);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  classical.scalars[op.getResult()] = combine(*lhs, *rhs);
  return success();
}

template <typename OpTy, typename Shift>
static LogicalResult applyShiftIndex(OpTy op, ClassicalEnv& classical,
                                     Shift shift) {
  if (!isa<IndexType>(op.getType())) {
    return op.emitError() << "QCO DD simulation only supports index "
                          << op.getOperationName();
  }
  auto lhs = lookupInteger(op.getLhs(), classical, op);
  auto rhs = lookupInteger(op.getRhs(), classical, op);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  if (rhs->isNegative() || rhs->uge(64)) {
    return op.emitError() << "shift amount out of range for QCO DD simulation";
  }
  classical.scalars[op.getResult()] =
      shift(*lhs, static_cast<unsigned>(rhs->getZExtValue()));
  return success();
}

static LogicalResult applyClassicalOp(Operation& op, ClassicalEnv& classical) {
  return TypeSwitch<Operation*, LogicalResult>(&op)
      .Case<arith::AndIOp>([&](arith::AndIOp andOp) -> LogicalResult {
        return applyBinaryInteger(
            andOp, classical,
            [](const llvm::APInt& a, const llvm::APInt& b) { return a & b; });
      })
      .Case<arith::OrIOp>([&](arith::OrIOp orOp) -> LogicalResult {
        return applyBinaryInteger(
            orOp, classical,
            [](const llvm::APInt& a, const llvm::APInt& b) { return a | b; });
      })
      .Case<arith::XOrIOp>([&](arith::XOrIOp xorOp) -> LogicalResult {
        return applyBinaryInteger(
            xorOp, classical,
            [](const llvm::APInt& a, const llvm::APInt& b) { return a ^ b; });
      })
      .Case<arith::AddIOp>([&](arith::AddIOp addOp) {
        return applyBinaryInteger(
            addOp, classical,
            [](const llvm::APInt& a, const llvm::APInt& b) { return a + b; },
            true);
      })
      .Case<arith::SubIOp>([&](arith::SubIOp subOp) {
        return applyBinaryInteger(
            subOp, classical,
            [](const llvm::APInt& a, const llvm::APInt& b) { return a - b; },
            true);
      })
      .Case<arith::MulIOp>([&](arith::MulIOp mulOp) {
        return applyBinaryInteger(
            mulOp, classical,
            [](const llvm::APInt& a, const llvm::APInt& b) { return a * b; },
            true);
      })
      .Case<arith::ShLIOp>([&](arith::ShLIOp shli) -> LogicalResult {
        return applyShiftIndex(shli, classical,
                               [](const llvm::APInt& value, unsigned amount) {
                                 return value.shl(amount);
                               });
      })
      .Case<arith::ShRUIOp>([&](arith::ShRUIOp shrui) -> LogicalResult {
        return applyShiftIndex(shrui, classical,
                               [](const llvm::APInt& value, unsigned amount) {
                                 return value.lshr(amount);
                               });
      })
      .Case<arith::CmpIOp>([&](arith::CmpIOp cmp) -> LogicalResult {
        auto lhs = lookupInteger(cmp.getLhs(), classical, cmp);
        auto rhs = lookupInteger(cmp.getRhs(), classical, cmp);
        if (failed(lhs) || failed(rhs)) {
          return failure();
        }
        classical.scalars[cmp.getResult()] = llvm::APInt(
            1, arith::applyCmpPredicate(cmp.getPredicate(), *lhs, *rhs));
        return success();
      })
      .Case<arith::SelectOp>([&](arith::SelectOp select) -> LogicalResult {
        auto cond = lookupBool(select.getCondition(), classical, select);
        if (failed(cond)) {
          return failure();
        }
        if (!select.getType().isInteger(1) &&
            !isa<IndexType>(select.getType())) {
          return select.emitError()
                 << "QCO DD simulation only supports select on i1 or index";
        }
        auto t = lookupInteger(select.getTrueValue(), classical, select);
        auto f = lookupInteger(select.getFalseValue(), classical, select);
        if (failed(t) || failed(f)) {
          return failure();
        }
        classical.scalars[select.getResult()] = *cond ? *t : *f;
        return success();
      })
      .Case<arith::IndexCastUIOp>([&](arith::IndexCastUIOp cast) {
        return applyI1ToIndex(cast.getIn(), cast.getOut(), cast, classical);
      })
      .Default([](Operation* unsupported) {
        return unsupported->emitError()
               << "unsupported classical op for QCO DD simulation: "
               << unsupported->getName().getStringRef();
      });
}

/// Bind each source SSA onto the corresponding dest (qubits via `QubitMap`,
/// CBit registers by sharing their storage, and scalar classical values via
/// `ClassicalEnv::bindFrom`). Callers must ensure equal sizes.
static LogicalResult bindValuePairs(ValueRange sources, ValueRange dests,
                                    WalkState& walk, Operation* op) {
  for (auto [src, dest] : llvm::zip_equal(sources, dests)) {
    if (isa<QubitType>(dest.getType())) {
      if (!isa<QubitType>(src.getType())) {
        return op->emitError()
               << "qubit/classical SSA type mismatch for QCO DD simulation";
      }
      const auto q = walk.qubits->lookup(src);
      if (!q) {
        return op->emitError()
               << "qubit SSA value is not mapped for QCO DD construction";
      }
      walk.qubits->bind(dest, *q);
    } else if (isa<cbit::RegisterType>(dest.getType())) {
      if (!isa<cbit::RegisterType>(src.getType()) ||
          src.getType() != dest.getType()) {
        return op->emitError() << "QCO DD simulation only supports matching "
                                  "CBit register values";
      }
      const auto it = walk.classical->registers.find(src);
      if (it == walk.classical->registers.end()) {
        return op->emitError()
               << "CBit register is not mapped for QCO DD simulation";
      }
      walk.classical->registers[dest] = it->second;
    } else if (failed(walk.classical->bindFrom(src, dest, op))) {
      return failure();
    }
  }
  return success();
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
static LogicalResult applyRegionBranch(ValueRange linearOperands, Block& block,
                                       WalkState& walk, StateDD& state,
                                       Operation* parent) {
  if (failed(
          bindValuePairs(linearOperands, block.getArguments(), walk, parent))) {
    return failure();
  }
  if (failed(walkBlock(block, walk, state))) {
    return failure();
  }
  auto yield = cast<YieldOp>(block.getTerminator());
  return bindValuePairs(yield.getOperands(), parent->getResults(), walk, yield);
}

template <typename StateDD>
static LogicalResult applyOp(Operation& op, WalkState& walk, StateDD& state) {
  return TypeSwitch<Operation*, LogicalResult>(&op)
      .template Case<StaticOp, SinkOp>([](auto) { return success(); })
      .template Case<arith::ConstantOp>([&](arith::ConstantOp constant) {
        return recordConstant(constant, *walk.classical);
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
      .template Case<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::AddIOp,
                     arith::SubIOp, arith::MulIOp, arith::ShLIOp,
                     arith::ShRUIOp, arith::CmpIOp, arith::SelectOp,
                     arith::IndexCastUIOp>([&](Operation* classicalOp) {
        return applyClassicalOp(*classicalOp, *walk.classical);
      })
      .template Case<func::ReturnOp>([&](func::ReturnOp returnOp) {
        return validateReturn(returnOp, *walk.qubits);
      })
      .template Case<MeasureOp>([&](MeasureOp measureOp) -> LogicalResult {
        if constexpr (!std::is_same_v<StateDD, dd::VectorDD>) {
          return measureOp.emitError()
                 << "measurements are not supported for QCO DD functionality "
                    "construction";
        } else {
          if (walk.rng == nullptr && !walk.deferTerminalMeasurements) {
            return measureOp.emitError()
                   << "measurements require simulate(..., rng)";
          }
          const auto q = walk.qubits->lookup(measureOp.getQubitIn());
          if (!q) {
            return measureOp.emitError()
                   << "qubit SSA value is not mapped for QCO DD construction";
          }
          if (walk.rng == nullptr) {
            walk.qubits->bind(measureOp.getQubitOut(), *q);
            return success();
          }
          const char bit = walk.dd->measureOneCollapsing(state, *q, *walk.rng);
          walk.classical->scalars[measureOp.getResult()] =
              llvm::APInt(1, bit == '1');
          if (walk.classicalBits != nullptr) {
            walk.classicalBits->push_back(bit);
          }
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
        if constexpr (!std::is_same_v<StateDD, dd::VectorDD>) {
          return ifOp.emitError()
                 << "control-flow is not supported for QCO DD functionality "
                    "construction";
        } else {
          const auto condIt = walk.classical->scalars.find(ifOp.getCondition());
          if (condIt == walk.classical->scalars.end()) {
            return ifOp.emitError()
                   << "if condition is not a concrete classical value";
          }
          Block* block =
              !condIt->second.isZero() ? ifOp.thenBlock() : ifOp.elseBlock();
          return applyRegionBranch(ifOp.getQubits(), *block, walk, state, ifOp);
        }
      })
      .template Case<IndexSwitchOp>(
          [&](IndexSwitchOp switchOp) -> LogicalResult {
            if constexpr (!std::is_same_v<StateDD, dd::VectorDD>) {
              return switchOp.emitError()
                     << "control-flow is not supported for QCO DD "
                        "functionality construction";
            } else {
              const auto idxIt =
                  walk.classical->scalars.find(switchOp.getArg());
              if (idxIt == walk.classical->scalars.end()) {
                return switchOp.emitError()
                       << "index_switch argument is not a concrete index";
              }
              const int64_t selector = idxIt->second.getSExtValue();
              const auto cases = switchOp.getCases();
              Block* block = switchOp.getDefaultBlock();
              for (auto [i, caseValue] : llvm::enumerate(cases)) {
                if (caseValue == selector) {
                  block = switchOp.getCaseBlock(i);
                  break;
                }
              }
              return applyRegionBranch(switchOp.getTargets(), *block, walk,
                                       state, switchOp);
            }
          })
      .template Case<scf::ForOp>([&](scf::ForOp forOp) -> LogicalResult {
        if constexpr (!std::is_same_v<StateDD, dd::VectorDD>) {
          return forOp.emitError()
                 << "scf.for is not supported for QCO DD functionality "
                    "construction";
        } else {
          auto lb = lookupIndex(forOp.getLowerBound(), *walk.classical, forOp);
          auto ub = lookupIndex(forOp.getUpperBound(), *walk.classical, forOp);
          auto step = lookupIndex(forOp.getStep(), *walk.classical, forOp);
          if (failed(lb) || failed(ub) || failed(step)) {
            return failure();
          }
          const bool unsignedCmp = forOp.getUnsignedCmp();
          const auto stepValue = static_cast<uint64_t>(*step);
          if (*step <= 0) {
            return forOp.emitError()
                   << "scf.for step must be positive for QCO DD simulation";
          }
          constexpr int64_t maxTrips = 10000;
          int64_t trips = 0;
          const bool hasTrips = unsignedCmp ? static_cast<uint64_t>(*ub) >
                                                  static_cast<uint64_t>(*lb)
                                            : *ub > *lb;
          if (hasTrips) {
            // Use unsigned arithmetic to avoid signed-overflow UB when
            // classical bounds are extreme (e.g. INT64_MIN / INT64_MAX).
            const auto span =
                static_cast<uint64_t>(*ub) - static_cast<uint64_t>(*lb);
            const uint64_t tripsU = ((span - 1) / stepValue) + 1;
            if (tripsU > static_cast<uint64_t>(maxTrips)) {
              return forOp.emitError()
                     << "scf.for trip count exceeds QCO DD simulation limit of "
                     << maxTrips;
            }
            trips = static_cast<int64_t>(tripsU);
          }

          Block& body = *forOp.getBody();
          SmallVector<Value> carried(forOp.getInits().begin(),
                                     forOp.getInits().end());

          for (int64_t t = 0; t < trips; ++t) {
            const auto offset =
                static_cast<uint64_t>(t) * static_cast<uint64_t>(*step);
            walk.classical->scalars[body.getArgument(0)] =
                llvm::APInt(64, static_cast<uint64_t>(*lb) + offset);
            auto iterArgs = body.getArguments().drop_front();
            if (failed(bindValuePairs(carried, iterArgs, walk, forOp))) {
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
        }
      })
      .template Case<func::CallOp>([&](func::CallOp call) -> LogicalResult {
        auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            call, call.getCalleeAttr());
        if (!callee) {
          return call.emitError() << "func.call callee '" << call.getCallee()
                                  << "' could not be resolved";
        }
        if (!callee.getBody().hasOneBlock()) {
          return call.emitError()
                 << "func.call callee must have a single-block body";
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

        // Walk the callee body without its terminator so entry-function-only
        // `validateReturn` (canonical wire order) is not applied to callees.
        if (failed(walkBlock(callee.getBody().front(), walk, state))) {
          return failure();
        }

        // Map callee return operands onto call results via the return op.
        auto returnOp =
            cast<func::ReturnOp>(callee.getBody().front().getTerminator());
        return bindValuePairs(returnOp.getOperands(), call.getResults(), walk,
                              call);
      })
      .template Case<CtrlOp>([&](CtrlOp ctrlOp) -> LogicalResult {
        if (auto inner = mqt::getSoleBodyUnitary<UnitaryOpInterface>(
                *ctrlOp.getBody())) {
          auto decoded = decodeStandardGate(inner);
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
            auto decoded = decodeStandardGate(unitary);
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
static LogicalResult walkFunction(func::FuncOp func, WalkState& walkState,
                                  StateDD& state) {
  // Function bodies include `func.return` as terminator; region walks skip
  // `qco.yield` and bind it separately.
  for (Operation& op : func.getBody().front()) {
    if (failed(applyOp(op, walkState, state))) {
      return failure();
    }
  }
  return success();
}

static FailureOr<QubitMap> prepare(func::FuncOp func, const dd::Package& dd) {
  if (!func.getBody().hasOneBlock()) {
    return func.emitError()
           << "QCO DD construction expects a single-block function body";
  }

  QubitMap qubits;
  for (StaticOp staticOp : func.getBody().front().getOps<StaticOp>()) {
    const auto q = static_cast<qc::Qubit>(staticOp.getIndex());
    qubits.bind(staticOp.getQubit(), q);
    qubits.numQubits = std::max(qubits.numQubits, static_cast<size_t>(q) + 1);
  }
  // No `qco.static`: treat qubit-typed block arguments as wires `0..n-1`.
  if (qubits.numQubits == 0) {
    qc::Qubit next = 0;
    for (Value arg : func.getArguments()) {
      if (!isa<QubitType>(arg.getType())) {
        continue;
      }
      qubits.bind(arg, next);
      qubits.numQubits =
          std::max(qubits.numQubits, static_cast<size_t>(next) + 1);
      ++next;
    }
  }
  if (dd.qubits() < qubits.numQubits) {
    return func.emitError() << "DD package has " << dd.qubits()
                            << " qubits but function uses " << qubits.numQubits;
  }
  return qubits;
}

FailureOr<dd::MatrixDD> buildFunctionality(func::FuncOp func, dd::Package& dd) {
  auto qubitsOr = prepare(func, dd);
  if (failed(qubitsOr)) {
    return failure();
  }
  QubitMap qubits = std::move(*qubitsOr);
  ClassicalEnv classical;
  WalkState walkState{.qubits = &qubits,
                      .classical = &classical,
                      .dd = &dd,
                      .rng = nullptr,
                      .classicalBits = nullptr};

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
             const bool deferTerminalMeasurements = false) {
  auto qubitsOr = prepare(func, dd);
  if (failed(qubitsOr)) {
    dd.decRef(in);
    return failure();
  }
  QubitMap qubits = std::move(*qubitsOr);
  ClassicalEnv classical;
  WalkState walkState{.qubits = &qubits,
                      .classical = &classical,
                      .dd = &dd,
                      .rng = rng,
                      .deferTerminalMeasurements = deferTerminalMeasurements,
                      .classicalBits = classicalBits};

  dd::VectorDD state = in;
  if (failed(walkFunction(func, walkState, state))) {
    dd.decRef(state);
    return failure();
  }
  return state;
}

FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd) {
  return simulateImpl(func, in, dd, nullptr, nullptr);
}

FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd, std::mt19937_64& rng) {
  return simulateImpl(func, in, dd, &rng, nullptr);
}

namespace {
struct SamplingRequirements {
  bool dynamic = false;
  bool measured = false;
};
} // namespace

[[nodiscard]] static SamplingRequirements
getSamplingRequirements(func::FuncOp func, const bool recordClassics,
                        DenseSet<Operation*>* visiting = nullptr) {
  DenseSet<Operation*> localVisiting;
  DenseSet<Operation*>& active =
      visiting != nullptr ? *visiting : localVisiting;
  Operation* funcOp = func.getOperation();
  if (!active.insert(funcOp).second) {
    // Recursive call cycle: treat as dynamic to avoid infinite recursion.
    return {.dynamic = true};
  }

  bool measured = false;
  const WalkResult walked = func.getBody().walk([&](Operation* op) {
    if (isa<ResetOp>(op)) {
      return WalkResult::interrupt();
    }
    if (isa<MeasureOp>(op)) {
      // Sampling classical outcomes requires executing even terminal
      // measurements once per shot.
      if (recordClassics) {
        return WalkResult::interrupt();
      }
      measured = true;
      return WalkResult::advance();
    }
    // Terminal measurements can be deferred to the repeated measureAll calls.
    // Any subsequent computation may observe their collapsed state or result.
    if (measured && !isa<SinkOp, func::ReturnOp>(op)) {
      return WalkResult::interrupt();
    }
    if (auto call = dyn_cast<func::CallOp>(op)) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || !callee.getBody().hasOneBlock()) {
        return WalkResult::interrupt();
      }
      const auto calleeRequirements =
          getSamplingRequirements(callee, recordClassics, &active);
      // A terminal measurement can be deferred only in the entry function.
      // Calls must execute it per shot because the caller may consume the
      // collapsed qubit or a returned classical result.
      if (calleeRequirements.dynamic || calleeRequirements.measured) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  active.erase(funcOp);
  return {.dynamic = walked.wasInterrupted(), .measured = measured};
}

static FailureOr<SampleResult> sampleImpl(func::FuncOp func,
                                          const dd::VectorDD& in,
                                          dd::Package& dd, const size_t shots,
                                          std::mt19937_64& rng,
                                          const bool recordClassics) {
  SampleResult result;
  if (shots == 0) {
    dd.decRef(in);
    return result;
  }

  if (!getSamplingRequirements(func, recordClassics).dynamic) {
    auto stateOr = simulateImpl(func, in, dd, nullptr, nullptr,
                                /*deferTerminalMeasurements=*/true);
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
    auto stateOr =
        simulateImpl(func, in, dd, &rng, recordClassics ? &classical : nullptr);
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
sample(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd,
       const size_t shots, std::mt19937_64& rng) {
  auto result = sampleImpl(func, in, dd, shots, rng, /*recordClassics=*/false);
  if (failed(result)) {
    return failure();
  }
  return std::move(result->shots);
}

FailureOr<SampleResult> sampleWithClassics(func::FuncOp func,
                                           const dd::VectorDD& in,
                                           dd::Package& dd, const size_t shots,
                                           std::mt19937_64& rng) {
  return sampleImpl(func, in, dd, shots, rng, /*recordClassics=*/true);
}
FailureOr<std::map<std::string, size_t>> sample(func::FuncOp func,
                                                dd::Package& dd,
                                                const size_t shots,
                                                std::mt19937_64& rng) {
  auto qubitsOr = prepare(func, dd);
  if (failed(qubitsOr)) {
    return failure();
  }
  const size_t n = qubitsOr->numQubits;
  return sample(func, dd::makeZeroState(n, dd), dd, shots, rng);
}

FailureOr<SampleResult> sampleWithClassics(func::FuncOp func, dd::Package& dd,
                                           const size_t shots,
                                           std::mt19937_64& rng) {
  auto qubitsOr = prepare(func, dd);
  if (failed(qubitsOr)) {
    return failure();
  }
  const size_t n = qubitsOr->numQubits;
  return sampleWithClassics(func, dd::makeZeroState(n, dd), dd, shots, rng);
}

} // namespace mlir::qco
