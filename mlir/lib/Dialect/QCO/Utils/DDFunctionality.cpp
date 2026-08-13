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
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
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
#include <map>
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
  DenseMap<Value, bool> bools;
  DenseMap<Value, int64_t> indices;

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
  QubitMap* qubits;
  ClassicalEnv* classical;
  dd::Package* dd;
  std::mt19937_64* rng = nullptr;
  bool deferTerminalMeasurements = false;
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
    decoded.params.push_back(static_cast<dd::fp>(*utils::valueToDouble(param)));
  }
  return std::optional{std::move(decoded)};
}

/// QCO matrices are MSB-first (operand 0 = high bit).
[[nodiscard]] static size_t qcoIndexFromDdIndex(const size_t ddIndex,
                                                const size_t numQubits) {
  const auto shift = static_cast<unsigned>(64 - numQubits);
  return llvm::reverseBits(ddIndex) >> shift;
}

[[nodiscard]] static dd::CMat toCMatInDdBasis(const DynamicMatrix& qcoMatrix,
                                              size_t numQubits) {
  const auto dim = static_cast<size_t>(qcoMatrix.rows());
  dd::CMat out(dim, dd::CVec(dim));
  for (size_t row = 0; row < dim; ++row) {
    for (size_t col = 0; col < dim; ++col) {
      out[row][col] =
          qcoMatrix(static_cast<int64_t>(qcoIndexFromDdIndex(row, numQubits)),
                    static_cast<int64_t>(qcoIndexFromDdIndex(col, numQubits)));
    }
  }
  return out;
}

/// Embed a k-qubit QCO/MSB matrix onto @p wires of an n-qubit register.
[[nodiscard]] static DynamicMatrix
embedLocalInNQubitMsb(const DynamicMatrix& local, size_t n,
                      ArrayRef<qc::Qubit> wires) {
  const size_t k = wires.size();
  const auto dimN = static_cast<int64_t>(size_t{1} << n);
  DynamicMatrix out(dimN);
  const auto dimNSz = static_cast<size_t>(dimN);
  auto bitAt = [](size_t idx, size_t nQ, size_t q) -> size_t {
    return (idx >> (nQ - 1 - q)) & 1U;
  };
  llvm::SmallDenseSet<qc::Qubit, 8> wireSet;
  wireSet.insert(wires.begin(), wires.end());
  for (size_t row = 0; row < dimNSz; ++row) {
    for (size_t col = 0; col < dimNSz; ++col) {
      bool idleMatch = true;
      for (size_t q = 0; q < n; ++q) {
        if (wireSet.contains(static_cast<qc::Qubit>(q))) {
          continue;
        }
        if (bitAt(row, n, q) != bitAt(col, n, q)) {
          idleMatch = false;
          break;
        }
      }
      if (!idleMatch) {
        continue;
      }
      size_t rLoc = 0;
      size_t cLoc = 0;
      for (size_t i = 0; i < k; ++i) {
        rLoc = (rLoc << 1) | bitAt(row, n, wires[i]);
        cLoc = (cLoc << 1) | bitAt(col, n, wires[i]);
      }
      out(static_cast<int64_t>(row), static_cast<int64_t>(col)) =
          local(static_cast<int64_t>(rLoc), static_cast<int64_t>(cLoc));
    }
  }
  return out;
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
    const auto theta = *utils::valueToDouble(gphase.getTheta());
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

  // Dense embed of a k-qubit QCO/MSB matrix into n wires, then rewrite to
  // DD/LSB. Cap at 12 qubits (~256 MiB dense `CMat`).
  if (walk.qubits->numQubits > 12) {
    return unitary.emitError()
           << "QCO DD matrix fallback supports at most 12 qubits";
  }

  DynamicMatrix embedded = local;
  const bool fullWidthCanonical =
      wires.size() == walk.qubits->numQubits &&
      llvm::all_of(llvm::enumerate(wires),
                   [](const auto& it) { return it.value() == it.index(); });
  if (!fullWidthCanonical) {
    embedded = embedLocalInNQubitMsb(local, walk.qubits->numQubits, wires);
  }

  state = walk.dd->applyOperation(walk.dd->makeDDFromMatrix(toCMatInDdBasis(
                                      embedded, walk.qubits->numQubits)),
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
    classical.bools[constant.getResult()] = attr.getValue() != 0;
  } else if (isa<IndexType>(constant.getType())) {
    classical.indices[constant.getResult()] = attr.getInt();
  }
  return success();
}

static LogicalResult applyIndexCastUI(arith::IndexCastUIOp cast,
                                      ClassicalEnv& classical) {
  Value in = cast.getIn();
  if (!in.getType().isInteger(1)) {
    return cast.emitError()
           << "QCO DD simulation only supports index_castui from i1";
  }
  const auto it = classical.bools.find(in);
  if (it == classical.bools.end()) {
    return cast.emitError()
           << "classical i1 SSA value is not mapped for QCO DD simulation";
  }
  classical.indices[cast.getOut()] = it->second ? 1 : 0;
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

template <typename OpTy>
static LogicalResult applyBinaryI1(OpTy op, ClassicalEnv& classical,
                                   bool (*combine)(bool, bool)) {
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

template <typename OpTy>
static LogicalResult applyClassicalBitwise(OpTy op, ClassicalEnv& classical) {
  if constexpr (std::is_same_v<OpTy, arith::ShLIOp>) {
    return applyBinaryIndex(op, classical,
                            [](int64_t a, int64_t b) { return a << b; });
  } else if (op.getType().isInteger(1)) {
    if constexpr (std::is_same_v<OpTy, arith::AndIOp>) {
      return applyBinaryI1(op, classical,
                           [](bool a, bool b) { return a && b; });
    } else if constexpr (std::is_same_v<OpTy, arith::OrIOp>) {
      return applyBinaryI1(op, classical,
                           [](bool a, bool b) { return a || b; });
    } else {
      return applyBinaryI1(op, classical,
                           [](bool a, bool b) { return a != b; });
    }
  } else if constexpr (std::is_same_v<OpTy, arith::AndIOp>) {
    return applyBinaryIndex(op, classical,
                            [](int64_t a, int64_t b) { return a & b; });
  } else if constexpr (std::is_same_v<OpTy, arith::OrIOp>) {
    return applyBinaryIndex(op, classical,
                            [](int64_t a, int64_t b) { return a | b; });
  } else {
    return applyBinaryIndex(op, classical,
                            [](int64_t a, int64_t b) { return a ^ b; });
  }
}

static LogicalResult bindLinearArgs(ValueRange operands, Block& block,
                                    WalkState& walk, Operation* op) {
  for (auto [operand, arg] : llvm::zip_equal(operands, block.getArguments())) {
    const auto q = walk.qubits->lookup(operand);
    if (!q) {
      return op->emitError()
             << "qubit SSA value is not mapped for QCO DD construction";
    }
    walk.qubits->bind(arg, *q);
  }
  return success();
}

static LogicalResult bindYieldResults(YieldOp yield,
                                      ValueRange classicalResults,
                                      ValueRange linearResults,
                                      WalkState& walk) {
  size_t idx = 0;
  for (Value result : classicalResults) {
    if (failed(
            walk.classical->bindFrom(yield.getOperand(idx++), result, yield))) {
      return failure();
    }
  }
  for (Value result : linearResults) {
    const auto q = walk.qubits->lookup(yield.getOperand(idx++));
    if (!q) {
      return yield.emitError()
             << "yielded qubit SSA value is not mapped for QCO DD construction";
    }
    walk.qubits->bind(result, *q);
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
static LogicalResult applyOp(Operation& op, WalkState& walk, StateDD& state) {
  return TypeSwitch<Operation*, LogicalResult>(&op)
      .template Case<StaticOp, SinkOp>([](auto) { return success(); })
      .template Case<arith::ConstantOp>([&](arith::ConstantOp constant) {
        return recordConstant(constant, *walk.classical);
      })
      .template Case<arith::IndexCastUIOp>([&](arith::IndexCastUIOp cast) {
        return applyIndexCastUI(cast, *walk.classical);
      })
      .template Case<arith::AndIOp>([&](arith::AndIOp classicalOp) {
        return applyClassicalBitwise(classicalOp, *walk.classical);
      })
      .template Case<arith::OrIOp>([&](arith::OrIOp classicalOp) {
        return applyClassicalBitwise(classicalOp, *walk.classical);
      })
      .template Case<arith::XOrIOp>([&](arith::XOrIOp classicalOp) {
        return applyClassicalBitwise(classicalOp, *walk.classical);
      })
      .template Case<arith::ShLIOp>([&](arith::ShLIOp classicalOp) {
        return applyClassicalBitwise(classicalOp, *walk.classical);
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
          if (walk.rng == nullptr) {
            if (walk.deferTerminalMeasurements) {
              const auto q = walk.qubits->lookup(measureOp.getQubitIn());
              if (!q) {
                return measureOp.emitError()
                       << "qubit SSA value is not mapped for QCO DD "
                          "construction";
              }
              walk.qubits->bind(measureOp.getQubitOut(), *q);
              return success();
            }
            return measureOp.emitError()
                   << "measurements require simulate(..., rng)";
          }
          const auto q = walk.qubits->lookup(measureOp.getQubitIn());
          if (!q) {
            return measureOp.emitError()
                   << "qubit SSA value is not mapped for QCO DD construction";
          }
          const char bit = walk.dd->measureOneCollapsing(state, *q, *walk.rng);
          walk.classical->bools[measureOp.getResult()] = bit == '1';
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
          const auto condIt = walk.classical->bools.find(ifOp.getCondition());
          if (condIt == walk.classical->bools.end()) {
            return ifOp.emitError()
                   << "if condition is not a concrete classical value";
          }
          Block* block = condIt->second ? ifOp.thenBlock() : ifOp.elseBlock();
          YieldOp yield = condIt->second ? ifOp.thenYield() : ifOp.elseYield();
          return applyRegionBranch(ifOp.getQubits(), *block, yield,
                                   ifOp.getClassicalResults(),
                                   ifOp.getLinearResults(), walk, state, ifOp);
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
                  walk.classical->indices.find(switchOp.getArg());
              if (idxIt == walk.classical->indices.end()) {
                return switchOp.emitError()
                       << "index_switch argument is not a concrete index";
              }
              const int64_t selector = idxIt->second;
              const auto cases = switchOp.getCases();
              Block* block = switchOp.getDefaultBlock();
              YieldOp yield = switchOp.getDefaultYield();
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
            }
          })
      .template Case<CtrlOp>([&](CtrlOp ctrlOp) -> LogicalResult {
        if (auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(
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
static LogicalResult walk(func::FuncOp func, WalkState& walkState,
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
  WalkState walkState{
      .qubits = &qubits, .classical = &classical, .dd = &dd, .rng = nullptr};

  dd::MatrixDD state =
      qubits.numQubits == 0
          ? dd::MatrixDD::one()
          : dd.createInitialMatrix(std::vector<bool>(qubits.numQubits, false));
  if (failed(walk(func, walkState, state))) {
    if (qubits.numQubits != 0) {
      dd.decRef(state);
    }
    return failure();
  }
  return state;
}

static FailureOr<dd::VectorDD>
simulateImpl(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd,
             std::mt19937_64* rng,
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
                      .deferTerminalMeasurements = deferTerminalMeasurements};

  dd::VectorDD state = in;
  if (failed(walk(func, walkState, state))) {
    dd.decRef(state);
    return failure();
  }
  return state;
}

FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd) {
  return simulateImpl(func, in, dd, nullptr);
}

FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd, std::mt19937_64& rng) {
  return simulateImpl(func, in, dd, &rng);
}

[[nodiscard]] static bool requiresDynamicSampling(func::FuncOp func) {
  bool dynamic = false;
  bool measured = false;
  func.walk([&](Operation* op) {
    if (isa<ResetOp, IfOp, IndexSwitchOp>(op)) {
      dynamic = true;
      return WalkResult::interrupt();
    }
    if (isa<MeasureOp>(op)) {
      measured = true;
      return WalkResult::advance();
    }
    // Terminal measurements can be deferred to the repeated measureAll calls.
    // Any subsequent computation may observe their collapsed state or result.
    if (measured && !isa<SinkOp, func::ReturnOp>(op)) {
      dynamic = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return dynamic;
}

FailureOr<std::map<std::string, size_t>>
sample(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd,
       const size_t shots, std::mt19937_64& rng) {
  std::map<std::string, size_t> counts;
  if (shots == 0) {
    dd.decRef(in);
    return counts;
  }

  if (!requiresDynamicSampling(func)) {
    auto stateOr = simulateImpl(func, in, dd, nullptr,
                                /*deferTerminalMeasurements=*/true);
    if (failed(stateOr)) {
      return failure();
    }
    dd::VectorDD state = *stateOr;
    for (size_t i = 0; i < shots; ++i) {
      counts[dd.measureAll(state, false, rng)] += 1;
    }
    dd.decRef(state);
    return counts;
  }

  for (size_t i = 0; i < shots; ++i) {
    dd.incRef(in);
    auto stateOr = simulateImpl(func, in, dd, &rng);
    if (failed(stateOr)) {
      dd.decRef(in);
      return failure();
    }
    dd::VectorDD state = *stateOr;
    counts[dd.measureAll(state, false, rng)] += 1;
    dd.decRef(state);
  }
  dd.decRef(in);
  return counts;
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

} // namespace mlir::qco
