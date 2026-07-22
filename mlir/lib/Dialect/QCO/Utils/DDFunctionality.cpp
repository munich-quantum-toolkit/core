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
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "ir/Definitions.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
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

struct DecodedGate {
  qc::OpType type = qc::OpType::None;
  std::vector<dd::fp> params;
};

} // namespace

/// `std::nullopt` if @p op is not a standard gate; failure on non-constant
/// parameters.
static FailureOr<std::optional<DecodedGate>> decodeStandardGate(Operation* op) {
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
  auto unitary = cast<UnitaryOpInterface>(op);
  for (Value param : unitary.getParameters()) {
    const auto value = utils::valueToDouble(param);
    if (!value) {
      return op->emitError()
             << "gate parameters must be compile-time constants";
    }
    decoded.params.push_back(static_cast<dd::fp>(*value));
  }
  return std::optional{std::move(decoded)};
}

/// QCO matrices are MSB-first (operand 0 = high bit); DD is LSB-first.
[[nodiscard]] static dd::CMat toCMatInDdBasis(const DynamicMatrix& qcoMatrix,
                                              size_t numQubits) {
  const auto dim = static_cast<size_t>(qcoMatrix.rows());
  const auto shift = static_cast<unsigned>(64 - numQubits);
  dd::CMat out(dim, dd::CVec(dim));
  for (size_t row = 0; row < dim; ++row) {
    for (size_t col = 0; col < dim; ++col) {
      const auto qcoRow = llvm::reverseBits(row) >> shift;
      const auto qcoCol = llvm::reverseBits(col) >> shift;
      out[row][col] =
          qcoMatrix(static_cast<int64_t>(qcoRow), static_cast<int64_t>(qcoCol));
    }
  }
  return out;
}

template <typename StateDD>
static LogicalResult applyUnitaryMatrix(UnitaryOpInterface unitary,
                                        QubitMap& qubits, dd::Package& dd,
                                        StateDD& state) {
  Operation* op = unitary.getOperation();
  if (auto gphase = dyn_cast<GPhaseOp>(op)) {
    const auto theta = utils::valueToDouble(gphase.getTheta());
    if (!theta) {
      return unitary.emitError()
             << "unitary must have a compile-time constant matrix";
    }
    auto id = dd::Package::makeIdent();
    id.w = dd.cn.lookup(std::cos(*theta), std::sin(*theta));
    state = dd.applyOperation(id, state);
    return success();
  }
  if (isa<BarrierOp>(op)) {
    return qubits.remapUnitary(unitary);
  }

  DynamicMatrix local;
  if (!unitary.getUnitaryMatrixDynamic(local)) {
    return unitary.emitError()
           << "unitary must have a compile-time constant matrix";
  }

  auto wiresOr = qubits.lookupRange(unitary.getInputQubits(), op);
  if (failed(wiresOr)) {
    return failure();
  }
  const ArrayRef<qc::Qubit> wires = *wiresOr;

  if (wires.size() == 1) {
    const dd::GateMatrix mat{local(0, 0), local(0, 1), local(1, 0),
                             local(1, 1)};
    state = dd.applyOperation(dd.makeGateDD(mat, wires[0]), state);
    return qubits.remapUnitary(unitary);
  }

  if (wires.size() == 2) {
    dd::TwoQubitGateMatrix mat{};
    for (size_t row = 0; row < mat.size(); ++row) {
      for (size_t col = 0; col < mat[row].size(); ++col) {
        mat[row][col] = local(static_cast<int64_t>(row),
                              static_cast<int64_t>(col));
      }
    }
    state = dd.applyOperation(
        dd.makeTwoQubitGateDD(mat, wires[0], wires[1]), state);
    return qubits.remapUnitary(unitary);
  }

  // Map full-width matrices from QCO/MSB order to DD/LSB. Cap at 12 qubits
  // (~256 MiB dense `CMat`).
  if (qubits.numQubits > 12) {
    return unitary.emitError()
           << "QCO DD matrix fallback supports at most 12 qubits";
  }

  if (wires.size() != qubits.numQubits ||
      !llvm::all_of(llvm::enumerate(wires), [](const auto& it) {
        return it.value() == it.index();
      })) {
    return op->emitError()
           << "QCO DD matrix fallback supports full-width unitaries on qubits "
              "0..n-1";
  }

  state = dd.applyOperation(
      dd.makeDDFromMatrix(toCMatInDdBasis(local, qubits.numQubits)), state);
  return qubits.remapUnitary(unitary);
}

template <typename StateDD>
static LogicalResult
applyDecodedStandard(UnitaryOpInterface unitary, const DecodedGate& gate,
                     const qc::Controls& controls, QubitMap& qubits,
                     dd::Package& dd, StateDD& state) {
  SmallVector<Value> targetVals;
  for (size_t i = 0; i < unitary.getNumTargets(); ++i) {
    targetVals.push_back(unitary.getInputTarget(i));
  }
  auto targets = qubits.lookupRange(targetVals, unitary.getOperation());
  if (failed(targets)) {
    return failure();
  }
  state = dd.applyOperation(
      getStandardOperationDD(dd, gate.type, gate.params, controls,
                             {targets->begin(), targets->end()}),
      state);
  return qubits.remapUnitary(unitary);
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

template <typename StateDD>
static LogicalResult applyOp(Operation& op, QubitMap& qubits, dd::Package& dd,
                             StateDD& state) {
  return TypeSwitch<Operation*, LogicalResult>(&op)
      .template Case<StaticOp, SinkOp, arith::ConstantOp>(
          [](auto) { return success(); })
      .template Case<func::ReturnOp>(
          [&](func::ReturnOp returnOp) {
            return validateReturn(returnOp, qubits);
          })
      .template Case<CtrlOp>([&](CtrlOp ctrlOp) -> LogicalResult {
        if (auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(
                *ctrlOp.getBody())) {
          auto decoded = decodeStandardGate(inner.getOperation());
          if (failed(decoded)) {
            return failure();
          }
          if (*decoded) {
            auto controlQubits =
                qubits.lookupRange(ctrlOp.getControlsIn(), ctrlOp);
            if (failed(controlQubits)) {
              return failure();
            }
            qc::Controls controls;
            for (qc::Qubit q : *controlQubits) {
              controls.emplace(q);
            }
            return applyDecodedStandard(ctrlOp, **decoded, controls, qubits, dd,
                                        state);
          }
        }
        return applyUnitaryMatrix(ctrlOp, qubits, dd, state);
      })
      .template Case<UnitaryOpInterface>(
          [&](UnitaryOpInterface unitary) -> LogicalResult {
            auto decoded = decodeStandardGate(&op);
            if (failed(decoded)) {
              return failure();
            }
            if (*decoded) {
              return applyDecodedStandard(unitary, **decoded, {}, qubits, dd,
                                          state);
            }
            return applyUnitaryMatrix(unitary, qubits, dd, state);
          })
      .Default([](Operation* unsupported) {
        return unsupported->emitError()
               << "unsupported op for QCO DD construction: "
               << unsupported->getName().getStringRef();
      });
}

template <typename StateDD>
static LogicalResult walk(func::FuncOp func, QubitMap& qubits, dd::Package& dd,
                          StateDD& state) {
  for (Operation& op : func.getBody().front()) {
    if (failed(applyOp(op, qubits, dd, state))) {
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

  dd::MatrixDD state =
      qubits.numQubits == 0
          ? dd::MatrixDD::one()
          : dd.createInitialMatrix(std::vector<bool>(qubits.numQubits, false));
  if (failed(walk(func, qubits, dd, state))) {
    if (qubits.numQubits != 0) {
      dd.decRef(state);
    }
    return failure();
  }
  return state;
}

FailureOr<dd::VectorDD> simulate(func::FuncOp func, const dd::VectorDD& in,
                                 dd::Package& dd) {
  auto qubitsOr = prepare(func, dd);
  if (failed(qubitsOr)) {
    return failure();
  }
  QubitMap qubits = std::move(*qubitsOr);

  dd::VectorDD state = in;
  if (failed(walk(func, qubits, dd, state))) {
    dd.decRef(state);
    return failure();
  }
  return state;
}

} // namespace mlir::qco
