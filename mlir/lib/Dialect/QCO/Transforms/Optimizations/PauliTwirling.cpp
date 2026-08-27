/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Utils/Modifiers.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <array>
#include <cstdint>
#include <numbers>
#include <random>

namespace mlir::qco {

#define GEN_PASS_DEF_PAULITWIRL2QGATES
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

enum class Pauli : uint8_t { I, X, Y, Z };

struct Twirl {
  Pauli beforeFirst;
  Pauli beforeSecond;
  Pauli afterFirst;
  Pauli afterSecond;
  bool correctPhase;

  constexpr Twirl(const Pauli preFirst, const Pauli preSecond,
                  const Pauli postFirst, const Pauli postSecond,
                  const bool phase)
      : beforeFirst(preFirst), beforeSecond(preSecond), afterFirst(postFirst),
        afterSecond(postSecond), correctPhase(phase) {}
};

// Rows are ordered by the pre-Pauli pair II, IX, ..., ZZ. Each table is
// derived from the corresponding QCO gate matrix and satisfies
// exp(i * phase) * post * gate * pre = gate.
constexpr std::array<Twirl, 16> CX_TWIRLS = {{
    {Pauli::I, Pauli::I, Pauli::I, Pauli::I, false},
    {Pauli::I, Pauli::X, Pauli::I, Pauli::X, false},
    {Pauli::I, Pauli::Y, Pauli::Z, Pauli::Y, false},
    {Pauli::I, Pauli::Z, Pauli::Z, Pauli::Z, false},
    {Pauli::X, Pauli::I, Pauli::X, Pauli::X, false},
    {Pauli::X, Pauli::X, Pauli::X, Pauli::I, false},
    {Pauli::X, Pauli::Y, Pauli::Y, Pauli::Z, false},
    {Pauli::X, Pauli::Z, Pauli::Y, Pauli::Y, true},
    {Pauli::Y, Pauli::I, Pauli::Y, Pauli::X, false},
    {Pauli::Y, Pauli::X, Pauli::Y, Pauli::I, false},
    {Pauli::Y, Pauli::Y, Pauli::X, Pauli::Z, true},
    {Pauli::Y, Pauli::Z, Pauli::X, Pauli::Y, false},
    {Pauli::Z, Pauli::I, Pauli::Z, Pauli::I, false},
    {Pauli::Z, Pauli::X, Pauli::Z, Pauli::X, false},
    {Pauli::Z, Pauli::Y, Pauli::I, Pauli::Y, false},
    {Pauli::Z, Pauli::Z, Pauli::I, Pauli::Z, false},
}};

constexpr std::array<Twirl, 16> CZ_TWIRLS = {{
    {Pauli::I, Pauli::I, Pauli::I, Pauli::I, false},
    {Pauli::I, Pauli::X, Pauli::Z, Pauli::X, false},
    {Pauli::I, Pauli::Y, Pauli::Z, Pauli::Y, false},
    {Pauli::I, Pauli::Z, Pauli::I, Pauli::Z, false},
    {Pauli::X, Pauli::I, Pauli::X, Pauli::Z, false},
    {Pauli::X, Pauli::X, Pauli::Y, Pauli::Y, false},
    {Pauli::X, Pauli::Y, Pauli::Y, Pauli::X, true},
    {Pauli::X, Pauli::Z, Pauli::X, Pauli::I, false},
    {Pauli::Y, Pauli::I, Pauli::Y, Pauli::Z, false},
    {Pauli::Y, Pauli::X, Pauli::X, Pauli::Y, true},
    {Pauli::Y, Pauli::Y, Pauli::X, Pauli::X, false},
    {Pauli::Y, Pauli::Z, Pauli::Y, Pauli::I, false},
    {Pauli::Z, Pauli::I, Pauli::Z, Pauli::I, false},
    {Pauli::Z, Pauli::X, Pauli::I, Pauli::X, false},
    {Pauli::Z, Pauli::Y, Pauli::I, Pauli::Y, false},
    {Pauli::Z, Pauli::Z, Pauli::Z, Pauli::Z, false},
}};

constexpr std::array<Twirl, 16> ECR_TWIRLS = {{
    {Pauli::I, Pauli::I, Pauli::I, Pauli::I, false},
    {Pauli::I, Pauli::X, Pauli::I, Pauli::X, false},
    {Pauli::I, Pauli::Y, Pauli::Z, Pauli::Z, true},
    {Pauli::I, Pauli::Z, Pauli::Z, Pauli::Y, false},
    {Pauli::X, Pauli::I, Pauli::Y, Pauli::X, true},
    {Pauli::X, Pauli::X, Pauli::Y, Pauli::I, true},
    {Pauli::X, Pauli::Y, Pauli::X, Pauli::Y, false},
    {Pauli::X, Pauli::Z, Pauli::X, Pauli::Z, false},
    {Pauli::Y, Pauli::I, Pauli::X, Pauli::X, true},
    {Pauli::Y, Pauli::X, Pauli::X, Pauli::I, true},
    {Pauli::Y, Pauli::Y, Pauli::Y, Pauli::Y, true},
    {Pauli::Y, Pauli::Z, Pauli::Y, Pauli::Z, true},
    {Pauli::Z, Pauli::I, Pauli::Z, Pauli::I, true},
    {Pauli::Z, Pauli::X, Pauli::Z, Pauli::X, true},
    {Pauli::Z, Pauli::Y, Pauli::I, Pauli::Z, false},
    {Pauli::Z, Pauli::Z, Pauli::I, Pauli::Y, true},
}};

constexpr std::array<Twirl, 16> ISWAP_TWIRLS = {{
    {Pauli::I, Pauli::I, Pauli::I, Pauli::I, false},
    {Pauli::I, Pauli::X, Pauli::Y, Pauli::Z, false},
    {Pauli::I, Pauli::Y, Pauli::X, Pauli::Z, true},
    {Pauli::I, Pauli::Z, Pauli::Z, Pauli::I, false},
    {Pauli::X, Pauli::I, Pauli::Z, Pauli::Y, false},
    {Pauli::X, Pauli::X, Pauli::X, Pauli::X, false},
    {Pauli::X, Pauli::Y, Pauli::Y, Pauli::X, false},
    {Pauli::X, Pauli::Z, Pauli::I, Pauli::Y, false},
    {Pauli::Y, Pauli::I, Pauli::Z, Pauli::X, true},
    {Pauli::Y, Pauli::X, Pauli::X, Pauli::Y, false},
    {Pauli::Y, Pauli::Y, Pauli::Y, Pauli::Y, false},
    {Pauli::Y, Pauli::Z, Pauli::I, Pauli::X, true},
    {Pauli::Z, Pauli::I, Pauli::I, Pauli::Z, false},
    {Pauli::Z, Pauli::X, Pauli::Y, Pauli::I, false},
    {Pauli::Z, Pauli::Y, Pauli::X, Pauli::I, true},
    {Pauli::Z, Pauli::Z, Pauli::Z, Pauli::Z, false},
}};

} // namespace

[[nodiscard]] static const std::array<Twirl, 16>* getTwirlTable(Operation* op) {
  if (auto ctrl = dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return nullptr;
    }
    auto inner = mqt::getSoleBodyUnitary<UnitaryOpInterface>(*ctrl.getBody());
    if (!inner) {
      return nullptr;
    }
    if (isa<XOp>(inner.getOperation())) {
      return &CX_TWIRLS;
    }
    if (isa<ZOp>(inner.getOperation())) {
      return &CZ_TWIRLS;
    }
    return nullptr;
  }
  if (isa<ECROp>(op)) {
    return &ECR_TWIRLS;
  }
  if (isa<iSWAPOp>(op)) {
    return &ISWAP_TWIRLS;
  }
  return nullptr;
}

static Value createPauli(IRRewriter& rewriter, const Location loc, Pauli pauli,
                         Value qubit) {
  switch (pauli) {
  case Pauli::I:
    return IdOp::create(rewriter, loc, qubit).getOutputQubit(0);
  case Pauli::X:
    return XOp::create(rewriter, loc, qubit).getOutputQubit(0);
  case Pauli::Y:
    return YOp::create(rewriter, loc, qubit).getOutputQubit(0);
  case Pauli::Z:
    return ZOp::create(rewriter, loc, qubit).getOutputQubit(0);
  }
  llvm_unreachable("unknown Pauli gate");
}

static void twirlGate(IRRewriter& rewriter, UnitaryOpInterface gate,
                      const Twirl& twirl) {
  auto* op = gate.getOperation();
  auto firstIn = gate.getInputQubit(0);
  auto secondIn = gate.getInputQubit(1);
  auto firstOut = gate.getOutputQubit(0);
  auto secondOut = gate.getOutputQubit(1);

  rewriter.setInsertionPoint(op);
  auto newFirstIn =
      createPauli(rewriter, op->getLoc(), twirl.beforeFirst, firstIn);
  auto newSecondIn =
      createPauli(rewriter, op->getLoc(), twirl.beforeSecond, secondIn);
  rewriter.modifyOpInPlace(op, [&]() {
    op->setOperand(0, newFirstIn);
    op->setOperand(1, newSecondIn);
  });

  rewriter.setInsertionPointAfter(op);
  auto newFirstOut =
      createPauli(rewriter, op->getLoc(), twirl.afterFirst, firstOut);
  auto newSecondOut =
      createPauli(rewriter, op->getLoc(), twirl.afterSecond, secondOut);
  rewriter.replaceAllUsesExcept(firstOut, newFirstOut,
                                newFirstOut.getDefiningOp());
  rewriter.replaceAllUsesExcept(secondOut, newSecondOut,
                                newSecondOut.getDefiningOp());

  if (twirl.correctPhase) {
    GPhaseOp::create(rewriter, op->getLoc(), std::numbers::pi);
  }
}

namespace {

struct PauliTwirl2QGates final
    : impl::PauliTwirl2QGatesBase<PauliTwirl2QGates> {
  using PauliTwirl2QGatesBase::PauliTwirl2QGatesBase;

protected:
  void runOnOperation() override {
    struct Candidate {
      UnitaryOpInterface gate;
      const std::array<Twirl, 16>* table;
    };
    SmallVector<Candidate> gates;
    getOperation().walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (const auto* table = getTwirlTable(op)) {
        gates.push_back({.gate = cast<UnitaryOpInterface>(op), .table = table});
      }
      return isa<CtrlOp, InvOp, PowOp>(op) ? WalkResult::skip()
                                           : WalkResult::advance();
    });

    IRRewriter rewriter(&getContext());
    std::mt19937_64 rng(seed);
    for (auto& [gate, table] : gates) {
      twirlGate(rewriter, gate, (*table)[rng() & 0xFU]);
    }
  }
};

} // namespace

} // namespace mlir::qco
