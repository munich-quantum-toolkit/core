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
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

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
  QubitMap& qubits;
  ClassicalEnv& classical;
  dd::Package& dd;
  std::mt19937_64* rng = nullptr;
  std::string* classicalBits = nullptr;
  DenseSet<Operation*>* activeCalls = nullptr;
};

} // namespace

/// `std::nullopt` if @p unitary is not a standard gate; failure if its unitary
/// matrix is not known at compile time.
static FailureOr<std::optional<DecodedGate>>
decodeStandardGate(UnitaryOpInterface unitary) {
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
  const size_t dimNSz = static_cast<size_t>(dimN);
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
    id.w = walk.dd.cn.lookup(std::cos(theta), std::sin(theta));
    state = walk.dd.applyOperation(id, state);
    return success();
  }
  if (isa<BarrierOp>(op)) {
    return walk.qubits.remapUnitary(unitary);
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

  if (wires.size() == 1) {
    const dd::GateMatrix mat{local(0, 0), local(0, 1), local(1, 0),
                             local(1, 1)};
    state = walk.dd.applyOperation(walk.dd.makeGateDD(mat, wires[0]), state);
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
    state = walk.dd.applyOperation(
        walk.dd.makeTwoQubitGateDD(mat, wires[0], wires[1]), state);
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
    state = walk.dd.applyOperation(
        walk.dd.makeThreeQubitGateDD(mat, wires[0], wires[1], wires[2]), state);
    return walk.qubits.remapUnitary(unitary);
  }

  // Dense embed of a k-qubit QCO/MSB matrix into n wires, then rewrite to
  // DD/LSB. Cap at 12 qubits (~256 MiB dense `CMat`).
  if (walk.qubits.numQubits > 12) {
    return unitary.emitError()
           << "QCO DD matrix fallback supports at most 12 qubits";
  }

  DynamicMatrix embedded = local;
  const bool fullWidthCanonical =
      wires.size() == walk.qubits.numQubits &&
      llvm::all_of(llvm::enumerate(wires),
                   [](const auto& it) { return it.value() == it.index(); });
  if (!fullWidthCanonical) {
    embedded = embedLocalInNQubitMsb(local, walk.qubits.numQubits, wires);
  }

  state = walk.dd.applyOperation(walk.dd.makeDDFromMatrix(toCMatInDdBasis(
                                     embedded, walk.qubits.numQubits)),
                                 state);
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
  state = walk.dd.applyOperation(
      getStandardOperationDD(walk.dd, gate.type, gate.params, controls,
                             {targets->begin(), targets->end()}),
      state);
  return walk.qubits.remapUnitary(unitary);
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
  if (!isa<IndexType>(cast.getType())) {
    return cast.emitError()
           << "QCO DD simulation only supports index_castui to index";
  }
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

static LogicalResult applyClassicalOp(Operation& op, ClassicalEnv& classical) {
  return TypeSwitch<Operation*, LogicalResult>(&op)
      .Case<arith::AndIOp>([&](arith::AndIOp andOp) -> LogicalResult {
        if (andOp.getType().isInteger(1)) {
          return applyBinaryI1(andOp, classical,
                               [](bool a, bool b) { return a && b; });
        }
        return applyBinaryIndex(andOp, classical,
                                [](int64_t a, int64_t b) { return a & b; });
      })
      .Case<arith::OrIOp>([&](arith::OrIOp orOp) -> LogicalResult {
        if (orOp.getType().isInteger(1)) {
          return applyBinaryI1(orOp, classical,
                               [](bool a, bool b) { return a || b; });
        }
        return applyBinaryIndex(orOp, classical,
                                [](int64_t a, int64_t b) { return a | b; });
      })
      .Case<arith::XOrIOp>([&](arith::XOrIOp xorOp) -> LogicalResult {
        if (xorOp.getType().isInteger(1)) {
          return applyBinaryI1(xorOp, classical,
                               [](bool a, bool b) { return a != b; });
        }
        return applyBinaryIndex(xorOp, classical,
                                [](int64_t a, int64_t b) { return a ^ b; });
      })
      .Case<arith::AddIOp>([&](arith::AddIOp addOp) {
        return applyBinaryIndex(addOp, classical,
                                [](int64_t a, int64_t b) { return a + b; });
      })
      .Case<arith::SubIOp>([&](arith::SubIOp subOp) {
        return applyBinaryIndex(subOp, classical,
                                [](int64_t a, int64_t b) { return a - b; });
      })
      .Case<arith::MulIOp>([&](arith::MulIOp mulOp) {
        return applyBinaryIndex(mulOp, classical,
                                [](int64_t a, int64_t b) { return a * b; });
      })
      .Case<arith::ShLIOp>([&](arith::ShLIOp shli) -> LogicalResult {
        if (!isa<IndexType>(shli.getType())) {
          return shli.emitError() << "QCO DD simulation only supports index "
                                  << shli.getOperationName();
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
        classical.indices[shli.getResult()] = *lhs
                                              << static_cast<unsigned>(*rhs);
        return success();
      })
      .Case<arith::ShRUIOp>([&](arith::ShRUIOp shrui) -> LogicalResult {
        if (!isa<IndexType>(shrui.getType())) {
          return shrui.emitError() << "QCO DD simulation only supports index "
                                   << shrui.getOperationName();
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
          lhs = static_cast<int64_t>(*lb);
          rhs = static_cast<int64_t>(*rb);
        } else {
          return cmp.emitError()
                 << "QCO DD simulation only supports cmpi on i1 or index";
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
        return select.emitError()
               << "QCO DD simulation only supports select on i1 or index";
      })
      .Case<arith::ExtUIOp>([&](arith::ExtUIOp ext) -> LogicalResult {
        if (!isa<IndexType>(ext.getType())) {
          return ext.emitError()
                 << "QCO DD simulation only supports extui to index";
        }
        Value in = ext.getIn();
        if (!in.getType().isInteger(1)) {
          return ext.emitError()
                 << "QCO DD simulation only supports extui from i1";
        }
        auto bit = lookupBool(in, classical, ext);
        if (failed(bit)) {
          return failure();
        }
        classical.indices[ext.getOut()] = *bit ? 1 : 0;
        return success();
      })
      .Case<arith::TruncIOp>([&](arith::TruncIOp trunc) -> LogicalResult {
        if (!trunc.getType().isInteger(1)) {
          return trunc.emitError()
                 << "QCO DD simulation only supports trunci to i1";
        }
        Value in = trunc.getIn();
        if (!isa<IndexType>(in.getType())) {
          return trunc.emitError()
                 << "QCO DD simulation only supports trunci from index";
        }
        auto idx = lookupIndex(in, classical, trunc);
        if (failed(idx)) {
          return failure();
        }
        classical.bools[trunc.getOut()] = (*idx & 1) != 0;
        return success();
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
    if (!isa<QubitType>(arg.getType())) {
      return op->emitError()
             << "QCO DD simulation does not support qtensor linear region "
                "args (qubits only)";
    }
    const auto q = walk.qubits.lookup(operand);
    if (!q) {
      return op->emitError()
             << "qubit SSA value is not mapped for QCO DD construction";
    }
    walk.qubits.bind(arg, *q);
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
    if (!isa<QubitType>(result.getType())) {
      return yield.emitError()
             << "QCO DD simulation does not support qtensor linear results "
                "(qubits only)";
    }
    const auto q = walk.qubits.lookup(yield.getOperand(idx++));
    if (!q) {
      return yield.emitError()
             << "yielded qubit SSA value is not mapped for QCO DD construction";
    }
    walk.qubits.bind(result, *q);
  }
  return success();
}

template <typename StateDD>
static LogicalResult applyOp(Operation& op, WalkState& walk, StateDD& state);

template <typename StateDD>
static LogicalResult walkFunction(func::FuncOp func, WalkState& walkState,
                                  StateDD& state);

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
        return recordConstant(constant, walk.classical);
      })
      .template Case<arith::IndexCastUIOp>([&](arith::IndexCastUIOp cast) {
        return applyIndexCastUI(cast, walk.classical);
      })
      .template Case<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::AddIOp,
                     arith::SubIOp, arith::MulIOp, arith::ShLIOp,
                     arith::ShRUIOp, arith::CmpIOp, arith::SelectOp,
                     arith::ExtUIOp, arith::TruncIOp>(
          [&](Operation* classicalOp) {
            return applyClassicalOp(*classicalOp, walk.classical);
          })
      .template Case<func::ReturnOp>([&](func::ReturnOp returnOp) {
        return validateReturn(returnOp, walk.qubits);
      })
      .template Case<MeasureOp>([&](MeasureOp measureOp) -> LogicalResult {
        if constexpr (!std::is_same_v<StateDD, dd::VectorDD>) {
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
          const char bit = walk.dd.measureOneCollapsing(state, *q, *walk.rng);
          walk.classical.bools[measureOp.getResult()] = bit == '1';
          if (walk.classicalBits != nullptr) {
            walk.classicalBits->push_back(bit);
          }
          walk.qubits.bind(measureOp.getQubitOut(), *q);
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
          const auto q = walk.qubits.lookup(resetOp.getQubitIn());
          if (!q) {
            return resetOp.emitError()
                   << "qubit SSA value is not mapped for QCO DD construction";
          }
          const char bit = walk.dd.measureOneCollapsing(state, *q, *walk.rng);
          if (bit == '1') {
            state = walk.dd.applyOperation(
                walk.dd.makeGateDD(dd::opToSingleQubitGateMatrix(qc::OpType::X),
                                   *q),
                state);
          }
          walk.qubits.bind(resetOp.getQubitOut(), *q);
          return success();
        }
      })
      .template Case<IfOp>([&](IfOp ifOp) -> LogicalResult {
        if constexpr (!std::is_same_v<StateDD, dd::VectorDD>) {
          return ifOp.emitError()
                 << "control-flow is not supported for QCO DD functionality "
                    "construction";
        } else {
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
        }
      })
      .template Case<IndexSwitchOp>(
          [&](IndexSwitchOp switchOp) -> LogicalResult {
            if constexpr (!std::is_same_v<StateDD, dd::VectorDD>) {
              return switchOp.emitError()
                     << "control-flow is not supported for QCO DD "
                        "functionality construction";
            } else {
              const auto idxIt = walk.classical.indices.find(switchOp.getArg());
              if (idxIt == walk.classical.indices.end()) {
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
      .template Case<scf::ForOp>([&](scf::ForOp forOp) -> LogicalResult {
        if constexpr (!std::is_same_v<StateDD, dd::VectorDD>) {
          return forOp.emitError()
                 << "scf.for is not supported for QCO DD functionality "
                    "construction";
        } else {
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
          const int64_t trips = (*ub > *lb) ? ((*ub - *lb - 1) / *step) + 1 : 0;
          constexpr int64_t maxTrips = 10000;
          if (trips > maxTrips) {
            return forOp.emitError()
                   << "scf.for trip count exceeds QCO DD simulation limit of "
                   << maxTrips;
          }

          Block& body = *forOp.getBody();
          SmallVector<Value> carried(forOp.getInits().begin(),
                                     forOp.getInits().end());

          auto bindCarriedToArgs = [&](ValueRange args) -> LogicalResult {
            if (carried.size() != args.size()) {
              return forOp.emitError()
                     << "scf.for iter_args size mismatch during simulation";
            }
            for (auto [src, arg] : llvm::zip_equal(carried, args)) {
              if (isa<QubitType>(arg.getType())) {
                if (!isa<QubitType>(src.getType())) {
                  return forOp.emitError()
                         << "scf.for iter_arg type mismatch (expected qubit)";
                }
                const auto q = walk.qubits.lookup(src);
                if (!q) {
                  return forOp.emitError()
                         << "qubit SSA value is not mapped for QCO DD "
                            "construction";
                }
                walk.qubits.bind(arg, *q);
              } else if (failed(walk.classical.bindFrom(src, arg, forOp))) {
                return failure();
              }
            }
            return success();
          };

          auto bindCarriedToResults = [&](ValueRange results) -> LogicalResult {
            if (carried.size() != results.size()) {
              return forOp.emitError()
                     << "scf.for result size mismatch during simulation";
            }
            for (auto [src, result] : llvm::zip_equal(carried, results)) {
              if (isa<QubitType>(result.getType())) {
                const auto q = walk.qubits.lookup(src);
                if (!q) {
                  return forOp.emitError()
                         << "qubit SSA value is not mapped for QCO DD "
                            "construction";
                }
                walk.qubits.bind(result, *q);
              } else if (failed(walk.classical.bindFrom(src, result, forOp))) {
                return failure();
              }
            }
            return success();
          };

          if (trips == 0) {
            return bindCarriedToResults(forOp.getResults());
          }

          for (int64_t t = 0; t < trips; ++t) {
            walk.classical.indices[body.getArgument(0)] = *lb + (t * *step);
            if (failed(bindCarriedToArgs(body.getArguments().drop_front()))) {
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
          return bindCarriedToResults(forOp.getResults());
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
        if (!callee.getBody().hasOneBlock()) {
          return call.emitError()
                 << "func.call callee must have a single-block body";
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

        if (call.getArgOperands().size() != callee.getNumArguments()) {
          walk.activeCalls->erase(calleeOp);
          return call.emitError()
                 << "func.call operand count does not match callee arguments";
        }
        for (auto [operand, arg] :
             llvm::zip_equal(call.getArgOperands(), callee.getArguments())) {
          if (isa<QubitType>(arg.getType())) {
            const auto q = walk.qubits.lookup(operand);
            if (!q) {
              walk.activeCalls->erase(calleeOp);
              return call.emitError()
                     << "qubit SSA value is not mapped for QCO DD construction";
            }
            walk.qubits.bind(arg, *q);
          } else if (failed(walk.classical.bindFrom(operand, arg, call))) {
            walk.activeCalls->erase(calleeOp);
            return failure();
          }
        }

        const LogicalResult walked = walkFunction(callee, walk, state);
        if (failed(walked)) {
          walk.activeCalls->erase(calleeOp);
          return failure();
        }

        // Map callee return operands onto call results via the return op.
        auto returnOp =
            dyn_cast<func::ReturnOp>(callee.getBody().front().getTerminator());
        if (!returnOp) {
          walk.activeCalls->erase(calleeOp);
          return call.emitError() << "callee missing func.return";
        }
        if (returnOp.getNumOperands() != call.getNumResults()) {
          walk.activeCalls->erase(calleeOp);
          return call.emitError()
                 << "func.call result count does not match callee return";
        }
        for (auto [retOperand, result] :
             llvm::zip_equal(returnOp.getOperands(), call.getResults())) {
          if (isa<QubitType>(result.getType())) {
            const auto q = walk.qubits.lookup(retOperand);
            if (!q) {
              walk.activeCalls->erase(calleeOp);
              return call.emitError()
                     << "returned qubit SSA value is not mapped for QCO DD "
                        "construction";
            }
            walk.qubits.bind(result, *q);
          } else if (failed(
                         walk.classical.bindFrom(retOperand, result, call))) {
            walk.activeCalls->erase(calleeOp);
            return failure();
          }
        }
        walk.activeCalls->erase(calleeOp);
        return success();
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
  DenseSet<Operation*> activeCalls;
  WalkState walkState{.qubits = qubits,
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
             std::mt19937_64* rng, std::string* classicalBits) {
  auto qubitsOr = prepare(func, dd);
  if (failed(qubitsOr)) {
    dd.decRef(in);
    return failure();
  }
  QubitMap qubits = std::move(*qubitsOr);
  ClassicalEnv classical;
  DenseSet<Operation*> activeCalls;
  WalkState walkState{.qubits = qubits,
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
                                 dd::Package& dd) {
  return simulateImpl(func, in, dd, nullptr, nullptr);
}

FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd, std::mt19937_64& rng) {
  return simulateImpl(func, in, dd, &rng, nullptr);
}

[[nodiscard]] static bool requiresDynamicSampling(func::FuncOp func) {
  bool dynamic = false;
  func.walk([&](Operation* op) {
    // Only stochastic collapse forces per-shot re-simulation. Deterministic
    // control-flow can reuse a single simulated state.
    if (isa<MeasureOp, ResetOp>(op)) {
      dynamic = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return dynamic;
}

FailureOr<std::map<std::string, std::size_t>>
sample(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd,
       const std::size_t shots, std::mt19937_64& rng) {
  std::map<std::string, std::size_t> counts;
  if (shots == 0) {
    dd.decRef(in);
    return counts;
  }

  if (!requiresDynamicSampling(func)) {
    auto stateOr = simulateImpl(func, in, dd, nullptr, nullptr);
    if (failed(stateOr)) {
      return failure();
    }
    dd::VectorDD state = *stateOr;
    for (std::size_t i = 0; i < shots; ++i) {
      counts[dd.measureAll(state, false, rng)] += 1;
    }
    dd.decRef(state);
    return counts;
  }

  for (std::size_t i = 0; i < shots; ++i) {
    dd.incRef(in);
    auto stateOr = simulateImpl(func, in, dd, &rng, nullptr);
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

FailureOr<SampleResult>
sampleWithClassics(func::FuncOp func, const dd::VectorDD& in, dd::Package& dd,
                   const std::size_t shots, std::mt19937_64& rng) {
  SampleResult result;
  if (shots == 0) {
    dd.decRef(in);
    return result;
  }

  if (!requiresDynamicSampling(func)) {
    auto stateOr = simulateImpl(func, in, dd, nullptr, nullptr);
    if (failed(stateOr)) {
      return failure();
    }
    dd::VectorDD state = *stateOr;
    for (std::size_t i = 0; i < shots; ++i) {
      result.shots[dd.measureAll(state, false, rng)] += 1;
    }
    dd.decRef(state);
    return result;
  }

  for (std::size_t i = 0; i < shots; ++i) {
    dd.incRef(in);
    std::string classical;
    auto stateOr = simulateImpl(func, in, dd, &rng, &classical);
    if (failed(stateOr)) {
      dd.decRef(in);
      return failure();
    }
    dd::VectorDD state = *stateOr;
    result.shots[dd.measureAll(state, false, rng)] += 1;
    if (!classical.empty()) {
      result.classical[classical] += 1;
    }
    dd.decRef(state);
  }
  dd.decRef(in);
  return result;
}

FailureOr<std::map<std::string, std::size_t>> sample(func::FuncOp func,
                                                     dd::Package& dd,
                                                     const std::size_t shots,
                                                     std::mt19937_64& rng) {
  auto qubitsOr = prepare(func, dd);
  if (failed(qubitsOr)) {
    return failure();
  }
  const size_t n = qubitsOr->numQubits;
  return sample(func, dd::makeZeroState(n, dd), dd, shots, rng);
}

FailureOr<SampleResult> sampleWithClassics(func::FuncOp func, dd::Package& dd,
                                           const std::size_t shots,
                                           std::mt19937_64& rng) {
  auto qubitsOr = prepare(func, dd);
  if (failed(qubitsOr)) {
    return failure();
  }
  const size_t n = qubitsOr->numQubits;
  return sampleWithClassics(func, dd::makeZeroState(n, dd), dd, shots, rng);
}

} // namespace mlir::qco
