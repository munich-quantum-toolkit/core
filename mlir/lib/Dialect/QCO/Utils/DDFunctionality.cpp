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
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Verifier.h>
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
  struct RegisterBit {
    std::optional<bool> value;
    std::optional<qc::Qubit> deferredWire;
  };
  using RegisterState = std::vector<RegisterBit>;

  DenseMap<Value, Attribute> scalars;
  DenseMap<Value, qc::Qubit> deferredMeasurements;
  /// Shared storage preserves CBit register identity across `func.call`.
  DenseMap<Value, std::shared_ptr<RegisterState>> registers;
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
  const DenseSet<Operation*>* deferredMeasurements = nullptr;
  size_t remainingExecutionSteps = 10'000;
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

static void bindInteger(Value result, const llvm::APInt& value,
                        ClassicalEnv& classical) {
  classical.scalars[result] = IntegerAttr::get(result.getType(), value);
}

static FailureOr<llvm::APInt>
lookupInteger(Value value, ClassicalEnv& classical, Operation* op) {
  const auto it = classical.scalars.find(value);
  if (it == classical.scalars.end()) {
    return op->emitError() << "classical SSA value is not mapped for QCO DD "
                              "simulation: "
                           << value.getType();
  }
  const auto attr = dyn_cast<IntegerAttr>(it->second);
  if (!attr) {
    return op->emitError()
           << "classical SSA value is not an integer for QCO DD simulation";
  }
  return attr.getValue();
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

static LogicalResult applyUnsignedIndexCast(Value in, Value out, Operation* op,
                                            ClassicalEnv& classical) {
  auto value = lookupInteger(in, classical, op);
  if (failed(value)) {
    return failure();
  }
  const auto integerType = dyn_cast<IntegerType>(out.getType());
  const unsigned width = integerType ? integerType.getWidth() : 64U;
  bindInteger(out, value->zextOrTrunc(width), classical);
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
  bindInteger(load.getResult(), llvm::APInt(1, *cell.value ? 1 : 0), classical);
  return success();
}

static LogicalResult applyBinaryInteger(Operation& op,
                                        ClassicalEnv& classical) {
  auto lhs = lookupInteger(op.getOperand(0), classical, &op);
  auto rhs = lookupInteger(op.getOperand(1), classical, &op);
  if (failed(lhs) || failed(rhs)) {
    return failure();
  }
  llvm::APInt result = *lhs;
  if (isa<arith::AndIOp>(&op)) {
    result &= *rhs;
  } else if (isa<arith::OrIOp>(&op)) {
    result |= *rhs;
  } else if (isa<arith::XOrIOp>(&op)) {
    result ^= *rhs;
  } else if (isa<arith::AddIOp>(&op)) {
    result += *rhs;
  } else if (isa<arith::SubIOp>(&op)) {
    result -= *rhs;
  } else if (isa<arith::MulIOp>(&op)) {
    result *= *rhs;
  } else {
    if (rhs->isNegative() || rhs->uge(lhs->getBitWidth())) {
      return op.emitError()
             << "shift amount out of range for QCO DD simulation";
    }
    const auto amount = static_cast<unsigned>(rhs->getZExtValue());
    result = isa<arith::ShLIOp>(&op) ? lhs->shl(amount) : lhs->lshr(amount);
  }
  bindInteger(op.getResult(0), result, classical);
  return success();
}

static LogicalResult applyClassicalOp(Operation& op, ClassicalEnv& classical) {
  return TypeSwitch<Operation*, LogicalResult>(&op)
      .Case<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::AddIOp,
            arith::SubIOp, arith::MulIOp, arith::ShLIOp, arith::ShRUIOp>(
          [&](Operation* binary) {
            return applyBinaryInteger(*binary, classical);
          })
      .Case<arith::CmpIOp>([&](arith::CmpIOp cmp) -> LogicalResult {
        auto lhs = lookupInteger(cmp.getLhs(), classical, cmp);
        auto rhs = lookupInteger(cmp.getRhs(), classical, cmp);
        if (failed(lhs) || failed(rhs)) {
          return failure();
        }
        bindInteger(cmp.getResult(),
                    llvm::APInt(1, arith::applyCmpPredicate(cmp.getPredicate(),
                                                            *lhs, *rhs)),
                    classical);
        return success();
      })
      .Case<arith::SelectOp>([&](arith::SelectOp select) -> LogicalResult {
        auto cond = lookupBool(select.getCondition(), classical, select);
        if (failed(cond)) {
          return failure();
        }
        if (!isa<IntegerType, IndexType>(select.getType())) {
          return select.emitError()
                 << "QCO DD simulation only supports integer or index select";
        }
        auto t = lookupInteger(select.getTrueValue(), classical, select);
        auto f = lookupInteger(select.getFalseValue(), classical, select);
        if (failed(t) || failed(f)) {
          return failure();
        }
        bindInteger(select.getResult(), *cond ? *t : *f, classical);
        return success();
      })
      .Case<arith::IndexCastUIOp>([&](arith::IndexCastUIOp cast) {
        return applyUnsignedIndexCast(cast.getIn(), cast.getOut(), cast,
                                      classical);
      })
      .Default([](Operation* unsupported) {
        return unsupported->emitError()
               << "unsupported classical op for QCO DD simulation: "
               << unsupported->getName().getStringRef();
      });
}

static FailureOr<LoopRange> resolveLoop(scf::ForOp forOp,
                                        ClassicalEnv& classical,
                                        const size_t remainingSteps) {
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
                                    WalkState& walk, Operation* op) {
  const QubitMap sourceQubits = *walk.qubits;
  const ClassicalEnv sourceClassical = *walk.classical;
  for (auto [src, dest] : llvm::zip_equal(sources, dests)) {
    if (isa<QubitType>(dest.getType())) {
      if (!isa<QubitType>(src.getType())) {
        return op->emitError()
               << "qubit/classical SSA type mismatch for QCO DD simulation";
      }
      const auto q = sourceQubits.lookup(src);
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
      const auto it = sourceClassical.registers.find(src);
      if (it == sourceClassical.registers.end()) {
        return op->emitError()
               << "CBit register is not mapped for QCO DD simulation";
      }
      walk.classical->registers[dest] = it->second;
    } else {
      const auto value = sourceClassical.scalars.find(src);
      if (value == sourceClassical.scalars.end()) {
        return op->emitError()
               << "classical SSA value is not mapped for QCO DD simulation";
      }
      walk.classical->scalars[dest] = value->second;
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
      .template Case<StaticOp, AllocOp, SinkOp>([](auto) { return success(); })
      .template Case<arith::ConstantOp>([&](arith::ConstantOp constant) {
        if (auto attr = dyn_cast<IntegerAttr>(constant.getValue())) {
          walk.classical->scalars[constant.getResult()] = attr;
        }
        return success();
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
          bindInteger(measureOp.getResult(), llvm::APInt(1, bit == '1'),
                      *walk.classical);
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
          auto condition =
              lookupBool(ifOp.getCondition(), *walk.classical, ifOp);
          if (failed(condition)) {
            return failure();
          }
          Block* block = *condition ? ifOp.thenBlock() : ifOp.elseBlock();
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
              auto index =
                  lookupIndex(switchOp.getArg(), *walk.classical, switchOp);
              if (failed(index)) {
                return failure();
              }
              const auto cases = switchOp.getCases();
              Block* block = switchOp.getDefaultBlock();
              for (auto [i, caseValue] : llvm::enumerate(cases)) {
                if (caseValue == *index) {
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
            bindInteger(
                body.getArgument(0),
                range->induction.trunc(range->induction.getBitWidth() - 1),
                *walk.classical);
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
  Operation* verificationRoot = func.getOperation();
  if (auto moduleOp = func->getParentOfType<ModuleOp>()) {
    verificationRoot = moduleOp.getOperation();
  }
  if (failed(verify(verificationRoot))) {
    return failure();
  }
  if (!func.getBody().hasOneBlock()) {
    return func.emitError()
           << "QCO DD construction expects a single-block function body";
  }

  QubitMap qubits;
  DenseSet<qc::Qubit> staticQubits;
  for (StaticOp staticOp : func.getBody().front().getOps<StaticOp>()) {
    const auto q = static_cast<qc::Qubit>(staticOp.getIndex());
    if (!staticQubits.insert(q).second) {
      return staticOp.emitError()
             << "duplicate static qubit index " << staticOp.getIndex();
    }
    qubits.bind(staticOp.getQubit(), q);
    qubits.numQubits = std::max(qubits.numQubits, static_cast<size_t>(q) + 1);
  }
  if (qubits.numQubits == 0) {
    qc::Qubit next = 0;
    for (Value arg : func.getArguments()) {
      if (!isa<QubitType>(arg.getType())) {
        continue;
      }
      qubits.bind(arg, next++);
    }
    qubits.numQubits = next;
  }
  for (AllocOp alloc : func.getBody().front().getOps<AllocOp>()) {
    qubits.bind(alloc.getResult(), static_cast<qc::Qubit>(qubits.numQubits++));
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
  WalkState walkState{
      .qubits = &qubits, .classical = &classical, .dd = &dd, .rng = nullptr};

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
             const QubitMap& preparedQubits, std::mt19937_64* rng,
             const DenseSet<Operation*>* deferredMeasurements = nullptr,
             ClassicalEnv* finalClassical = nullptr) {
  const size_t inputQubits =
      in.isTerminal() ? 0U : static_cast<size_t>(in.p->v) + 1U;
  if (inputQubits < preparedQubits.numQubits) {
    dd.decRef(in);
    return func.emitError()
           << "input state has " << inputQubits << " qubits but function uses "
           << preparedQubits.numQubits;
  }
  QubitMap qubits = preparedQubits;
  ClassicalEnv classical;
  WalkState walkState{.qubits = &qubits,
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
                                 dd::Package& dd) {
  auto qubits = prepare(func, dd);
  if (failed(qubits)) {
    dd.decRef(in);
    return failure();
  }
  return simulateImpl(func, in, dd, *qubits, nullptr);
}

FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd, std::mt19937_64& rng) {
  auto qubits = prepare(func, dd);
  if (failed(qubits)) {
    dd.decRef(in);
    return failure();
  }
  return simulateImpl(func, in, dd, *qubits, &rng);
}

static bool isOutputOnlyRegister(Value reg, const ArrayRef<Value> outputs) {
  return llvm::is_contained(outputs, reg) &&
         llvm::all_of(reg.getUses(), [reg](const OpOperand& use) {
           if (auto store = dyn_cast<cbit::StoreOp>(use.getOwner())) {
             return store.getReg() == reg;
           }
           return isa<func::ReturnOp>(use.getOwner());
         });
}

static bool isDeferrableMeasurement(MeasureOp measure, Block* entry,
                                    const ArrayRef<Value> outputs) {
  return measure->getBlock() == entry &&
         llvm::all_of(measure.getQubitOut().getUses(),
                      [](const OpOperand& use) {
                        return isa<SinkOp, func::ReturnOp>(use.getOwner());
                      }) &&
         llvm::all_of(measure.getResult().getUses(), [&](const OpOperand& use) {
           auto store = dyn_cast<cbit::StoreOp>(use.getOwner());
           return store && isOutputOnlyRegister(store.getReg(), outputs);
         });
}

static void analyzeSampling(func::FuncOp func, Block* entry,
                            const ArrayRef<Value> outputs,
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
      if (isDeferrableMeasurement(measure, entry, outputs)) {
        plan.deferredMeasurements.insert(op);
      } else {
        plan.dynamic = true;
      }
    } else if (auto call = dyn_cast<func::CallOp>(op)) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || !callee.getBody().hasOneBlock()) {
        plan.dynamic = true;
      } else {
        analyzeSampling(callee, entry, outputs, active, plan);
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
  analyzeSampling(func, &entry, plan.outputs, active, plan);
  return plan;
}

static FailureOr<std::string> encodeOutcome(const ArrayRef<Value> outputs,
                                            const ClassicalEnv& classical,
                                            const StringRef basis,
                                            const size_t numQubits) {
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
      } else if (cell.deferredWire && basis.size() == numQubits &&
                 *cell.deferredWire < numQubits) {
        outcome.push_back(basis[numQubits - 1 - *cell.deferredWire]);
      } else {
        return emitError(value.getLoc())
               << "returned CBit register element " << index << " is undefined";
      }
    }
  }
  return outcome;
}

FailureOr<std::map<std::string, size_t>> sample(func::FuncOp func,
                                                dd::Package& dd,
                                                const size_t shots,
                                                std::mt19937_64& rng) {
  auto qubits = prepare(func, dd);
  if (failed(qubits)) {
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

  const size_t numQubits = qubits->numQubits;
  const auto record = [&](const ClassicalEnv& classical,
                          const StringRef basis) -> LogicalResult {
    auto outcome = encodeOutcome(plan->outputs, classical, basis, numQubits);
    if (failed(outcome)) {
      return failure();
    }
    ++counts[*outcome];
    return success();
  };

  if (!plan->dynamic) {
    ClassicalEnv classical;
    auto state =
        simulateImpl(func, dd::makeZeroState(numQubits, dd), dd, *qubits,
                     nullptr, &plan->deferredMeasurements, &classical);
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
                              *qubits, &rng, nullptr, &classical);
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
