/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/Modifiers.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h" // IWYU pragma: keep (Passes.h.inc)
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_DECOMPOSEMULTICONTROLLED
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

// The synthesis routines below are independent implementations derived from
// the publications cited at the respective algorithms.

enum class Hp24DirtyMode : uint8_t { OneDirty, TwoDirty };
enum class ControlledTarget : uint8_t { X, Z, Phase };

constexpr double K_PI = std::numbers::pi;
constexpr double K_PI8 = K_PI / 8.0;

class GateEmitter {
public:
  GateEmitter(OpBuilder& builder, Location loc, SmallVector<Value>& wires)
      : builder_(&builder), loc_(loc), wires_(&wires) {}

  // Single- and two-qubit primitives
  void h(size_t q) {
    setWire(q, HOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void x(size_t q) {
    setWire(q, XOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void p(size_t q, double theta) {
    setWire(q, POp::create(*builder_, loc_, wire(q), theta).getOutputQubit(0));
  }

  void ry(size_t q, Value theta) {
    setWire(q, RYOp::create(*builder_, loc_, wire(q), theta).getOutputQubit(0));
  }

  void rz(size_t q, Value theta) {
    setWire(q, RZOp::create(*builder_, loc_, wire(q), theta).getOutputQubit(0));
  }

  void t(size_t q) {
    setWire(q, TOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void tdg(size_t q) {
    setWire(q, TdgOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void cx(size_t control, size_t target) {
    auto ctrlOp = CtrlOp::create(
        *builder_, loc_, wire(control), wire(target),
        [&](Value targetArg) -> Value {
          return XOp::create(*builder_, loc_, targetArg).getOutputQubit(0);
        });
    setWire(control, ctrlOp.getControlsOut()[0]);
    setWire(target, ctrlOp.getTargetsOut()[0]);
  }

  void cp(size_t control, size_t target, double theta) {
    auto ctrlOp =
        CtrlOp::create(*builder_, loc_, wire(control), wire(target),
                       [&](Value targetArg) -> Value {
                         return POp::create(*builder_, loc_, targetArg, theta)
                             .getOutputQubit(0);
                       });
    setWire(control, ctrlOp.getControlsOut()[0]);
    setWire(target, ctrlOp.getTargetsOut()[0]);
  }

  void crz(size_t control, size_t target, double theta) {
    const double half = theta / 2.0;
    p(target, half);
    cx(control, target);
    p(target, -half);
    cx(control, target);
  }

  // Controlled-RX via RX(theta) = H RZ(theta) H, reusing crz.
  void crx(size_t control, size_t target, double theta) {
    h(target);
    crz(control, target, theta);
    h(target);
  }

  // Building blocks left as QCO ops (further lowered by min-qubits)
  void emitCcx(size_t c0, size_t c1, size_t target) {
    emitCtrl({c0, c1}, target, [](OpBuilder& builder, Location loc, Value arg) {
      return XOp::create(builder, loc, arg).getOutputQubit(0);
    });
  }

  void emitThreeControlledX(size_t c0, size_t c1, size_t c2, size_t target) {
    emitCtrl({c0, c1, c2}, target,
             [](OpBuilder& builder, Location loc, Value arg) {
               return XOp::create(builder, loc, arg).getOutputQubit(0);
             });
  }

  void emitRCCX(size_t c0, size_t c1, size_t target) {
    auto rccxOp =
        RCCXOp::create(*builder_, loc_, wire(c0), wire(c1), wire(target));
    setWire(c0, rccxOp.getOutputQubit(0));
    setWire(c1, rccxOp.getOutputQubit(1));
    setWire(target, rccxOp.getOutputQubit(2));
  }

  void ccp(double theta, size_t c0, size_t c1, size_t target) {
    emitCtrl({c0, c1}, target,
             [theta](OpBuilder& builder, Location loc, Value arg) {
               return POp::create(builder, loc, arg, theta).getOutputQubit(0);
             });
  }

  // Fully expanded elementary sequences
  // (relative-phase gadgets are expanded directly in the HP24 planner)
  void emitRCCXSequence(size_t c0, size_t c1, size_t target) {
    h(target);
    t(target);
    cx(c1, target);
    tdg(target);
    cx(c0, target);
    t(target);
    cx(c1, target);
    tdg(target);
    h(target);
  }

  void emitTwoControlledXSequence(size_t c0, size_t c1, size_t target) {
    h(target);
    cx(c1, target);
    tdg(target);
    cx(c0, target);
    t(target);
    cx(c1, target);
    tdg(target);
    cx(c0, target);
    t(c1);
    t(target);
    h(target);
    cx(c0, c1);
    t(c0);
    tdg(c1);
    cx(c0, c1);
  }

  void emitThreeControlledXSequence() {
    h(3);
    p(0, K_PI8);
    p(1, K_PI8);
    p(2, K_PI8);
    p(3, K_PI8);
    cx(0, 1);
    p(1, -K_PI8);
    cx(0, 1);
    cx(1, 2);
    p(2, -K_PI8);
    cx(0, 2);
    p(2, K_PI8);
    cx(1, 2);
    p(2, -K_PI8);
    cx(0, 2);
    cx(2, 3);
    p(3, -K_PI8);
    cx(1, 3);
    p(3, K_PI8);
    cx(2, 3);
    p(3, -K_PI8);
    cx(0, 3);
    p(3, K_PI8);
    cx(2, 3);
    p(3, -K_PI8);
    cx(1, 3);
    p(3, K_PI8);
    cx(2, 3);
    p(3, -K_PI8);
    cx(0, 3);
    h(3);
  }

private:
  void emitCtrl(ArrayRef<size_t> controls, size_t target,
                function_ref<Value(OpBuilder&, Location, Value)> body) {
    SmallVector<Value> controlValues;
    controlValues.reserve(controls.size());
    for (size_t control : controls) {
      controlValues.push_back(wire(control));
    }
    auto ctrlOp = CtrlOp::create(*builder_, loc_, controlValues, wire(target),
                                 [&](Value targetArg) -> Value {
                                   return body(*builder_, loc_, targetArg);
                                 });
    for (size_t i = 0; i < controls.size(); ++i) {
      setWire(controls[i], ctrlOp.getControlsOut()[i]);
    }
    setWire(target, ctrlOp.getTargetsOut()[0]);
  }

  [[nodiscard]] Value wire(size_t local) const { return (*wires_)[local]; }

  void setWire(size_t local, Value value) { (*wires_)[local] = value; }

  OpBuilder* builder_;
  Location loc_;
  SmallVector<Value>* wires_;
};

//===----------------------------------------------------------------------===//
// Circuit plan
//===----------------------------------------------------------------------===//

/// Plan-level op kinds.
enum class PlanOpKind : uint8_t {
  H,
  X,
  P,
  CX,
  CP,
  CCP,
  CRX,
  CRZ,
  CCX,
  CCCX,
  RCCX,
};

/// One plan op. `wires` are local indices for `lowerPlan`.
struct PlanOp {
  PlanOpKind kind{};
  SmallVector<size_t, 4> wires;
  double angle = 0.0;
};

/// Ordered plan ops lowered by `lowerPlan`.
struct CircuitPlan {
  SmallVector<PlanOp, 32> ops;

  void append(PlanOp op) { ops.push_back(std::move(op)); }
};

struct ControlledGateSpec {
  ControlledTarget gate;
  std::optional<double> theta;
};

struct BorrowedControlPartition {
  size_t k1; // ceil(k / 2)
  size_t k2; // floor(k / 2)
};

} // namespace

[[nodiscard]] static size_t estimateBorrowedHelperMcxOps(size_t numControls) {
  if (numControls <= 3) {
    return 1;
  }
  // Two passes: CCX + (n-3)*5 gadget + RCCX + (n-3)*5 gadget each.
  return 2 * (2 + (10 * (numControls - 3)));
}

[[nodiscard]] static size_t estimateIncrementerPartitionedOps(size_t n) {
  return (16 * n) + 4;
}

[[nodiscard]] static size_t
estimateBorrowedDirtyIncrementerOps(size_t n, Hp24DirtyMode dirtyMode,
                                    bool flagAdd) {
  const bool oneDirty = dirtyMode == Hp24DirtyMode::OneDirty;
  const size_t k = oneDirty ? (n + 1) / 2 : (n + 2) / 2;
  const size_t lowIncrementWidth = oneDirty ? k : (1 + n - k);
  const size_t incrementerOps =
      estimateIncrementerPartitionedOps(lowIncrementWidth);
  const size_t halfMcxOps = estimateBorrowedHelperMcxOps(k);
  const size_t highIncrementOps = estimateIncrementerPartitionedOps(k);
  return (2 * incrementerOps) + (2 * halfMcxOps) + highIncrementOps +
         (2 * (n - k)) + 4 + (flagAdd ? 0 : (2 * n));
}

static void remapPlanOpInPlace(PlanOp& op, ArrayRef<size_t> map) {
  for (size_t& w : op.wires) {
    assert(w < map.size() && "plan wire out of remap range");
    w = map[w];
  }
}

static void appendPlanOps(CircuitPlan& dest, CircuitPlan src) {
  dest.ops.reserve(dest.ops.size() + src.ops.size());
  for (PlanOp& op : src.ops) {
    dest.append(std::move(op));
  }
}

/// Lower every `PlanOp` in `plan` onto `emitter`, in order.
static void lowerPlan(GateEmitter& emitter, const CircuitPlan& plan) {
  for (const PlanOp& op : plan.ops) {
    switch (op.kind) {
    case PlanOpKind::H:
      emitter.h(op.wires[0]);
      break;
    case PlanOpKind::X:
      emitter.x(op.wires[0]);
      break;
    case PlanOpKind::P:
      emitter.p(op.wires[0], op.angle);
      break;
    case PlanOpKind::CX:
      emitter.cx(op.wires[0], op.wires[1]);
      break;
    case PlanOpKind::CP:
      emitter.cp(op.wires[0], op.wires[1], op.angle);
      break;
    case PlanOpKind::CCP:
      emitter.ccp(op.angle, op.wires[0], op.wires[1], op.wires[2]);
      break;
    case PlanOpKind::CRX:
      emitter.crx(op.wires[0], op.wires[1], op.angle);
      break;
    case PlanOpKind::CRZ:
      emitter.crz(op.wires[0], op.wires[1], op.angle);
      break;
    case PlanOpKind::CCX:
      emitter.emitCcx(op.wires[0], op.wires[1], op.wires[2]);
      break;
    case PlanOpKind::CCCX:
      emitter.emitThreeControlledX(op.wires[0], op.wires[1], op.wires[2],
                                   op.wires[3]);
      break;
    case PlanOpKind::RCCX:
      emitter.emitRCCX(op.wires[0], op.wires[1], op.wires[2]);
      break;
    }
  }
}

/// Append `src` into `dest`, remapping each wire `w` to `map[w]`.
static void appendRemapped(CircuitPlan& dest, CircuitPlan src,
                           ArrayRef<size_t> map) {
  dest.ops.reserve(dest.ops.size() + src.ops.size());
  for (PlanOp& op : src.ops) {
    remapPlanOpInPlace(op, map);
    dest.append(std::move(op));
  }
}

//===----------------------------------------------------------------------===//
// HP24 MCZ core (Huang & Palsberg, PACMPL 2024, doi:10.1145/3656436)
//===----------------------------------------------------------------------===//
// Phase-π core on all-ones; no clean helpers (borrow target / a control as
// dirty). Callers use `MCZ = core` and `MCX = H . core . H` on the target.

// HP24 §4.3 relative-phase Toffoli gadget (and its reverse-order adjoint).
static void appendGadget(CircuitPlan& plan, size_t q0, size_t q1, size_t q2,
                         bool invert) {
  const double quarterPi = K_PI / 4.0; // T = p(pi/4), Tdg = p(-pi/4)
  if (!invert) {
    plan.append({.kind = PlanOpKind::H, .wires = {q2}});
    plan.append({.kind = PlanOpKind::P, .wires = {q2}, .angle = quarterPi});
    plan.append({.kind = PlanOpKind::CX, .wires = {q0, q2}});
    plan.append({.kind = PlanOpKind::P, .wires = {q2}, .angle = -quarterPi});
    plan.append({.kind = PlanOpKind::CX, .wires = {q1, q2}});
    return;
  }
  plan.append({.kind = PlanOpKind::CX, .wires = {q1, q2}});
  plan.append({.kind = PlanOpKind::P, .wires = {q2}, .angle = quarterPi});
  plan.append({.kind = PlanOpKind::CX, .wires = {q0, q2}});
  plan.append({.kind = PlanOpKind::P, .wires = {q2}, .angle = -quarterPi});
  plan.append({.kind = PlanOpKind::H, .wires = {q2}});
}

// HP24 Fig. 5 carry / uncarry steps for the wide leaf incrementer.
static void appendCarry(CircuitPlan& plan, size_t carry, size_t hi, size_t lo) {
  plan.append({.kind = PlanOpKind::CX, .wires = {carry, lo}});
  plan.append({.kind = PlanOpKind::CX, .wires = {carry, hi}});
  plan.append({.kind = PlanOpKind::CCX, .wires = {hi, lo, carry}});
}

static void appendUncarry(CircuitPlan& plan, size_t carry, size_t hi,
                          size_t lo) {
  plan.append({.kind = PlanOpKind::CCX, .wires = {hi, lo, carry}});
  plan.append({.kind = PlanOpKind::CX, .wires = {carry, hi}});
  plan.append({.kind = PlanOpKind::CX, .wires = {hi, lo}});
}
// HP24 Eq. (2) borrowed-helper MCX (controls, target, then dirty helpers).
static CircuitPlan planBorrowedHelperMcx(size_t numControls) {
  CircuitPlan plan;
  if (numControls == 1) {
    plan.append({.kind = PlanOpKind::CX, .wires = {0, 1}});
    return plan;
  }
  if (numControls == 2) {
    plan.append({.kind = PlanOpKind::CCX, .wires = {0, 1, 2}});
    return plan;
  }
  if (numControls == 3) {
    plan.append({.kind = PlanOpKind::CCCX, .wires = {0, 1, 2, 3}});
    return plan;
  }

  plan.ops.reserve(estimateBorrowedHelperMcxOps(numControls));

  const size_t target = numControls;
  const size_t topControl = numControls - 1;
  const size_t firstHelper = numControls + 1;
  const size_t lastHelper = firstHelper + numControls - 3;

  for (size_t pass = 0; pass < 2; ++pass) {
    plan.append(
        {.kind = PlanOpKind::CCX, .wires = {topControl, lastHelper, target}});
    for (size_t i = numControls - 3; i-- > 0;) {
      appendGadget(plan, i + 2, firstHelper + i, firstHelper + i + 1, false);
    }
    plan.append({.kind = PlanOpKind::RCCX, .wires = {0, 1, firstHelper}});
    for (size_t i = 0; i < numControls - 3; ++i) {
      appendGadget(plan, i + 2, firstHelper + i, firstHelper + i + 1, true);
    }
  }
  return plan;
}

// Wide leaf incrementer `U^n_{+1}` using dirty helpers and a carry ladder.
static CircuitPlan planIncrementerPartitioned(size_t n) {
  CircuitPlan plan;
  plan.ops.reserve(estimateIncrementerPartitionedOps(n));
  const size_t lastRegister = n - 1;
  const size_t carry = n;

  const auto conditionOnCarry = [&] {
    plan.append({.kind = PlanOpKind::X, .wires = {carry}});
    for (size_t q = 0; q < n; ++q) {
      plan.append({.kind = PlanOpKind::CX, .wires = {carry, q}});
    }
    plan.append({.kind = PlanOpKind::X, .wires = {carry}});
  };
  const auto sweepUp = [&] {
    for (size_t i = 0; i < n - 1; ++i) {
      appendCarry(plan, carry, carry + 1 + i, i);
    }
  };
  const auto sweepDown = [&] {
    for (size_t i = n - 1; i-- > 0;) {
      appendUncarry(plan, carry, carry + 1 + i, i);
    }
  };
  const auto flipHelpers = [&] {
    for (size_t i = 0; i < n - 1; ++i) {
      plan.append({.kind = PlanOpKind::X, .wires = {carry + 1 + i}});
    }
  };

  conditionOnCarry();
  sweepUp();
  plan.append({.kind = PlanOpKind::CX, .wires = {carry, lastRegister}});
  sweepDown();
  flipHelpers();
  sweepUp();
  plan.append({.kind = PlanOpKind::CX, .wires = {carry, lastRegister}});
  sweepDown();
  flipHelpers();
  plan.append({.kind = PlanOpKind::X, .wires = {lastRegister}});
  conditionOnCarry();
  return plan;
}

// HP24 Fig. 6/8 partitioned incrementer. One-dirty borrows the target;
// two-dirty also borrows the top control. `flagAdd == false` yields `U_{-1}`
// (Eq. (7)).
static CircuitPlan planBorrowedDirtyIncrementer(size_t n, bool flagAdd,
                                                Hp24DirtyMode dirtyMode) {
  CircuitPlan plan;
  const bool oneDirty = dirtyMode == Hp24DirtyMode::OneDirty;
  const size_t numDirty = oneDirty ? 1 : 2;
  const size_t k = oneDirty ? (n + 1) / 2 : (n + 2) / 2;
  const size_t helper = n;
  const size_t helper2 = n + 1;
  const size_t lowIncrementWidth = oneDirty ? k : (1 + n - k);
  plan.ops.reserve(estimateBorrowedDirtyIncrementerOps(n, dirtyMode, flagAdd));

  const auto flipRegister = [&] {
    for (size_t q = 0; q < n; ++q) {
      plan.append({.kind = PlanOpKind::X, .wires = {q}});
    }
  };

  // Sub-incrementer over the low half: wire order [helper, high half, low half,
  // (helper2)]; the trailing helpers become the borrowed workspace of `U_{+1}`.
  SmallVector<size_t, 16> lowIncrementWires;
  lowIncrementWires.push_back(helper);
  for (size_t q = k; q < n; ++q) {
    lowIncrementWires.push_back(q);
  }
  for (size_t q = 0; q < k; ++q) {
    lowIncrementWires.push_back(q);
  }
  if (numDirty == 2) {
    lowIncrementWires.push_back(helper2);
  }

  // Half-register MCX: wire order [low half, helper, high half, (helper2)] with
  // the borrowed helper as its target.
  SmallVector<size_t, 16> halfMcxWires;
  for (size_t q = 0; q < k; ++q) {
    halfMcxWires.push_back(q);
  }
  halfMcxWires.push_back(helper);
  for (size_t q = k; q < n; ++q) {
    halfMcxWires.push_back(q);
  }
  if (numDirty == 2) {
    halfMcxWires.push_back(helper2);
  }

  const auto incrementLow = [&] {
    appendRemapped(plan, planIncrementerPartitioned(lowIncrementWidth),
                   lowIncrementWires);
  };
  const auto halfMcx = [&] {
    // Include the high half (and optional helper2) as dirty workspace.
    appendRemapped(plan, planBorrowedHelperMcx(k), halfMcxWires);
  };
  const auto fanOutHelper = [&] {
    for (size_t q = k; q < n; ++q) {
      plan.append({.kind = PlanOpKind::CX, .wires = {helper, q}});
    }
  };

  if (!flagAdd) {
    flipRegister();
  }

  incrementLow();
  plan.append({.kind = PlanOpKind::X, .wires = {helper}});
  fanOutHelper();
  halfMcx();
  incrementLow();
  plan.append({.kind = PlanOpKind::X, .wires = {helper}});
  halfMcx();
  fanOutHelper();
  appendPlanOps(plan, planIncrementerPartitioned(k));

  if (!flagAdd) {
    flipRegister();
  }
  return plan;
}

// HP24 Theorem 4.4: `C^{n-1}(p(π))` via dirty incrementer + phase ladder.
static CircuitPlan planHp24Core(size_t numControls) {
  // Narrower widths use the specialized or SP22 constructions.
  assert(numControls >= 33 && "HP24 requires at least 33 controls");
  const size_t n = numControls + 1;
  const auto dirtyMode =
      numControls % 2 == 1 ? Hp24DirtyMode::OneDirty : Hp24DirtyMode::TwoDirty;
  CircuitPlan plan;
  const size_t target = n - 1;
  const size_t topControl = n - 2;
  const size_t registerWidth =
      dirtyMode == Hp24DirtyMode::OneDirty ? numControls : numControls - 1;
  plan.ops.reserve(
      estimateBorrowedDirtyIncrementerOps(registerWidth, dirtyMode, true) +
      estimateBorrowedDirtyIncrementerOps(registerWidth, dirtyMode, false) +
      (2 * (numControls - 1)) + 1);

  if (dirtyMode == Hp24DirtyMode::OneDirty) {
    const auto increment = [&](bool add) {
      appendPlanOps(plan,
                    planBorrowedDirtyIncrementer(numControls, add, dirtyMode));
    };
    increment(true);
    double phi = -K_PI;
    for (size_t q = numControls - 1; q > 0; --q) {
      phi /= 2.0;
      plan.append({.kind = PlanOpKind::CP, .wires = {q, target}, .angle = phi});
    }
    increment(false);
    phi = K_PI;
    for (size_t q = numControls - 1; q > 0; --q) {
      phi /= 2.0;
      plan.append({.kind = PlanOpKind::CP, .wires = {q, target}, .angle = phi});
    }
    plan.append({.kind = PlanOpKind::CP, .wires = {0, target}, .angle = phi});
    return plan;
  }

  const auto increment = [&](bool add) {
    appendPlanOps(
        plan, planBorrowedDirtyIncrementer(numControls - 1, add, dirtyMode));
  };
  increment(true);
  double phi = -K_PI;
  for (size_t q = numControls - 2; q > 0; --q) {
    phi /= 2.0;
    plan.append({
        .kind = PlanOpKind::CCP,
        .wires = {q, topControl, target},
        .angle = phi,
    });
  }
  increment(false);
  phi = K_PI;
  for (size_t q = numControls - 2; q > 0; --q) {
    phi /= 2.0;
    plan.append({
        .kind = PlanOpKind::CCP,
        .wires = {q, topControl, target},
        .angle = phi,
    });
  }
  plan.append({
      .kind = PlanOpKind::CCP,
      .wires = {0, topControl, target},
      .angle = phi,
  });
  return plan;
}

//===----------------------------------------------------------------------===//
// Rewrite entry points (called by patterns)
//===----------------------------------------------------------------------===//

static SmallVector<Value> threeControlledWires(ValueRange controls,
                                               Value target) {
  SmallVector<Value> wires(controls.begin(), controls.end());
  wires.push_back(target);
  return wires;
}

static SmallVector<Value> synthesizeRCCX(OpBuilder& builder, Location loc,
                                         Value control0, Value control1,
                                         Value target) {
  SmallVector<Value> wires = {control0, control1, target};
  GateEmitter(builder, loc, wires).emitRCCXSequence(0, 1, 2);
  return wires;
}

static SmallVector<Value> synthesizeTwoControlled(OpBuilder& builder,
                                                  Location loc, Value control0,
                                                  Value control1, Value target,
                                                  ControlledTarget gate) {
  SmallVector<Value> wires = {control0, control1, target};
  GateEmitter emitter(builder, loc, wires);
  switch (gate) {
  case ControlledTarget::X:
    emitter.emitTwoControlledXSequence(0, 1, 2);
    break;
  case ControlledTarget::Z:
    emitter.h(2);
    emitter.emitTwoControlledXSequence(0, 1, 2);
    emitter.h(2);
    break;
  case ControlledTarget::Phase:
    // Non-±π phase stays on synthesizeMultiControlledPhase /
    // planMcpTwoControlled.
    llvm_unreachable("use synthesizeMultiControlledPhase for C²P");
  }
  return wires;
}

static SmallVector<Value>
synthesizeThreeControlled(OpBuilder& builder, Location loc, ValueRange controls,
                          Value target, ControlledTarget gate) {
  SmallVector<Value> wires = threeControlledWires(controls, target);
  GateEmitter emitter(builder, loc, wires);
  if (gate == ControlledTarget::X) {
    emitter.emitThreeControlledXSequence();
  } else {
    emitter.h(3);
    emitter.emitThreeControlledXSequence();
    emitter.h(3);
  }
  return wires;
}

// Barenco residual for up to three controls: an RCCX peel followed by CX.
static void appendMcpBarencoRelative(CircuitPlan& plan, double theta,
                                     size_t numControls, size_t target) {
  if (numControls == 1) {
    plan.append({.kind = PlanOpKind::CP, .wires = {0, target}, .angle = theta});
    return;
  }

  const size_t peeled = numControls - 1;
  const double half = theta / 2.0;

  const auto appendPeelMcx = [&] {
    if (peeled == 2) {
      plan.append({.kind = PlanOpKind::RCCX, .wires = {0, 1, 2}});
      return;
    }
    plan.append({.kind = PlanOpKind::CX, .wires = {0, 1}});
  };

  plan.append(
      {.kind = PlanOpKind::CP, .wires = {peeled, target}, .angle = half});
  appendPeelMcx();
  plan.append(
      {.kind = PlanOpKind::CP, .wires = {peeled, target}, .angle = -half});
  appendPeelMcx();
  appendMcpBarencoRelative(plan, half, peeled, target);
}

// Maslov relative-phase C^3(X) (arXiv:1508.03273 Fig. 4); `invert` = adjoint.
static void appendRelativePhaseC3X(CircuitPlan& plan, size_t c0, size_t c1,
                                   size_t c2, size_t t, bool invert) {
  const double q = K_PI / 4.0; // T = p(pi/4)
  const std::array<PlanOp, 18> ops = {
      {
          {.kind = PlanOpKind::H, .wires = {t}},
          {.kind = PlanOpKind::P, .wires = {t}, .angle = q},
          {.kind = PlanOpKind::CX, .wires = {c2, t}},
          {.kind = PlanOpKind::P, .wires = {t}, .angle = -q},
          {.kind = PlanOpKind::H, .wires = {t}},
          {.kind = PlanOpKind::CX, .wires = {c0, t}},
          {.kind = PlanOpKind::P, .wires = {t}, .angle = q},
          {.kind = PlanOpKind::CX, .wires = {c1, t}},
          {.kind = PlanOpKind::P, .wires = {t}, .angle = -q},
          {.kind = PlanOpKind::CX, .wires = {c0, t}},
          {.kind = PlanOpKind::P, .wires = {t}, .angle = q},
          {.kind = PlanOpKind::CX, .wires = {c1, t}},
          {.kind = PlanOpKind::P, .wires = {t}, .angle = -q},
          {.kind = PlanOpKind::H, .wires = {t}},
          {.kind = PlanOpKind::P, .wires = {t}, .angle = q},
          {.kind = PlanOpKind::CX, .wires = {c2, t}},
          {.kind = PlanOpKind::P, .wires = {t}, .angle = -q},
          {.kind = PlanOpKind::H, .wires = {t}},
      },
  };
  if (!invert) {
    for (const PlanOp& op : ops) {
      plan.append(op);
    }
    return;
  }
  for (size_t i = ops.size(); i-- > 0;) {
    PlanOp op = ops[i];
    if (op.kind == PlanOpKind::P) {
      op.angle = -op.angle;
    }
    plan.append(std::move(op));
  }
}

/// Ancilla-free `C^4(Z)` (Barenco √Z peels + Maslov relative-phase toggles).
/// Controls 0..3, target 4.
static CircuitPlan planMczRelativePhaseK4() {
  CircuitPlan plan;
  constexpr size_t t = 4;
  const double half = K_PI / 2.0;
  const double quarter = K_PI / 4.0;
  const double eighth = K_PI / 8.0;

  // Peel C(√Z) on (3,t) with relative C^3(X) toggle.
  plan.append({.kind = PlanOpKind::CP, .wires = {3, t}, .angle = half});
  appendRelativePhaseC3X(plan, 0, 1, 2, 3, /*invert=*/false);
  plan.append({.kind = PlanOpKind::CP, .wires = {3, t}, .angle = -half});
  appendRelativePhaseC3X(plan, 0, 1, 2, 3, /*invert=*/true);

  // Peel C^3(S) with RCCX toggle on wire 2.
  plan.append({.kind = PlanOpKind::CP, .wires = {2, t}, .angle = quarter});
  plan.append({.kind = PlanOpKind::RCCX, .wires = {0, 1, 2}});
  plan.append({.kind = PlanOpKind::CP, .wires = {2, t}, .angle = -quarter});
  plan.append({.kind = PlanOpKind::RCCX, .wires = {0, 1, 2}});

  // Peel C^2(P(π/4)) with CX toggle on wire 1.
  plan.append({.kind = PlanOpKind::CP, .wires = {1, t}, .angle = eighth});
  plan.append({.kind = PlanOpKind::CX, .wires = {0, 1}});
  plan.append({.kind = PlanOpKind::CP, .wires = {1, t}, .angle = -eighth});
  plan.append({.kind = PlanOpKind::CX, .wires = {0, 1}});
  plan.append({.kind = PlanOpKind::CP, .wires = {0, t}, .angle = eighth});
  return plan;
}

static CircuitPlan mczCoreForWidth(size_t numControls);

static SmallVector<Value>
synthesizeMultiControlled(OpBuilder& builder, Location loc, ValueRange controls,
                          Value target, ControlledTarget gate) {
  SmallVector<Value> wires(controls.begin(), controls.end());
  wires.push_back(target);

  const size_t targetIdx = controls.size();
  GateEmitter emitter(builder, loc, wires);
  const CircuitPlan plan = mczCoreForWidth(controls.size());
  if (gate == ControlledTarget::X) {
    emitter.h(targetIdx);
    lowerPlan(emitter, plan);
    emitter.h(targetIdx);
  } else {
    lowerPlan(emitter, plan);
  }
  return wires;
}

//===----------------------------------------------------------------------===//
// Multi-controlled phase synthesis
//===----------------------------------------------------------------------===//

// SP22 LDD for general-angle MCP at and above this width.
static constexpr size_t K_MCP_SP22_MIN_CONTROLS = 5;
// No-ancilla MCX/MCZ uses SP22 MCP(π) through this control count.
static constexpr size_t K_MCX_SP22_MAX_CONTROLS = 32;

static CircuitPlan planMcp(double theta, size_t numControls);

//===----------------------------------------------------------------------===//
// Vale multi-controlled phase
//===----------------------------------------------------------------------===//

/// Vale control split: top `ceil(k/2)`, bottom `floor(k/2)`.
static BorrowedControlPartition partitionControls(size_t numControls) {
  return {.k1 = (numControls + 1) / 2, .k2 = numControls / 2};
}

/// Synthesize a controlled Pauli rotation using X R(a) X = R(-a) for Y/Z.
static SmallVector<Value>
synthesizeMultiControlledRotation(OpBuilder& builder, Location loc,
                                  ValueRange controls, Value target,
                                  UnitaryOpInterface rotation) {
  const size_t numControls = controls.size();
  const auto [k1, k2] = partitionControls(numControls);
  SmallVector<Value> wires(controls);
  wires.push_back(target);
  GateEmitter emitter(builder, loc, wires);

  const auto halfMcx = [&](size_t begin, size_t count) {
    SmallVector<size_t> map;
    map.reserve(numControls + 1);
    for (size_t control = begin; control < begin + count; ++control) {
      map.push_back(control);
    }
    map.push_back(numControls);
    // The balanced split provides at least count - 2 dirty helpers. Each
    // exact MCX restores these opposite controls before the next rotation.
    for (size_t control = 0; control < numControls; ++control) {
      if (control < begin || control >= begin + count) {
        map.push_back(control);
      }
    }
    auto plan = planBorrowedHelperMcx(count);
    for (auto& op : plan.ops) {
      remapPlanOpInPlace(op, map);
    }
    return plan;
  };
  const CircuitPlan firstHalf = halfMcx(0, k1);
  const CircuitPlan secondHalf = halfMcx(k1, k2);

  auto quarter =
      arith::MulFOp::create(builder, loc, rotation.getParameters()[0],
                            mqt::constantFromScalar(builder, loc, 0.25));
  auto negativeQuarter = arith::NegFOp::create(builder, loc, quarter);
  const bool isY = isa<RYOp>(rotation.getOperation());
  const bool isX = isa<RXOp>(rotation.getOperation());
  const auto rotate = [&](Value angle) {
    if (isY) {
      emitter.ry(numControls, angle);
    } else {
      emitter.rz(numControls, angle);
    }
  };

  // RX(theta) = H RZ(theta) H. In either remaining axis, the four rotations
  // sum to theta exactly when both control halves are all ones, else to zero.
  if (isX) {
    emitter.h(numControls);
  }
  for (size_t repeat = 0; repeat < 2; ++repeat) {
    lowerPlan(emitter, firstHalf);
    rotate(negativeQuarter);
    lowerPlan(emitter, secondHalf);
    rotate(quarter);
  }
  if (isX) {
    emitter.h(numControls);
  }
  return wires;
}

// Vale + Barenco-relative residual at this MCP width.
static constexpr size_t K_MCP_VALE_RELATIVE_RESIDUAL_CONTROLS = 4;

/// Vale24 Fig. 7 shell (arXiv:2302.06377): alternate half-MCX with target
/// `p(±θ/4)`. Controls then target. Caller appends the residual.
/// Only three or four controls reach this shell, so each half uses CX or CCX.
static void appendValeFig7Shell(CircuitPlan& plan, double theta,
                                size_t numControls) {
  const size_t target = numControls;
  const auto [k1, k2] = partitionControls(numControls);
  const double quarter = theta / 4.0;
  const auto appendHalfMcx = [&](size_t begin, size_t count) {
    if (count == 1) {
      plan.append({.kind = PlanOpKind::CX, .wires = {begin, target}});
      return;
    }
    plan.append({.kind = PlanOpKind::CCX, .wires = {begin, begin + 1, target}});
  };
  appendHalfMcx(0, k1);
  plan.append({.kind = PlanOpKind::P, .wires = {target}, .angle = -quarter});
  appendHalfMcx(k1, k2);
  plan.append({.kind = PlanOpKind::P, .wires = {target}, .angle = quarter});
  appendHalfMcx(0, k1);
  plan.append({.kind = PlanOpKind::P, .wires = {target}, .angle = -quarter});
  appendHalfMcx(k1, k2);
  plan.append({.kind = PlanOpKind::P, .wires = {target}, .angle = quarter});
}

/// Vale Fig. 7 + recursive `planMcp(θ/2)` residual on the control register.
static CircuitPlan planMcpVale(double theta, size_t numControls) {
  CircuitPlan plan;
  // Fig. 7 shell (8 ops) + optimized C²P residual at the production width k=3.
  plan.ops.reserve(18);
  appendValeFig7Shell(plan, theta, numControls);
  appendPlanOps(plan, planMcp(theta / 2.0, numControls - 1));
  return plan;
}

/// Vale shell with Barenco-relative (RCCX) residual.
static CircuitPlan planMcpValeRelativeResidual(double theta,
                                               size_t numControls) {
  CircuitPlan plan;
  appendValeFig7Shell(plan, theta, numControls);
  appendMcpBarencoRelative(plan, theta / 2.0, numControls - 1, numControls - 1);
  return plan;
}

/// Optimized ancilla-free `C^2(P(θ))`. Wires: `c0`, `c1`, target.
static CircuitPlan planMcpTwoControlled(double theta) {
  CircuitPlan plan;
  plan.ops.reserve(10);
  const double quarter = theta / 4.0;
  const double half = theta / 2.0;
  constexpr size_t c0 = 0;
  constexpr size_t c1 = 1;
  constexpr size_t target = 2;
  plan.append({.kind = PlanOpKind::CX, .wires = {c0, target}});
  plan.append({.kind = PlanOpKind::P, .wires = {target}, .angle = -quarter});
  plan.append({.kind = PlanOpKind::CX, .wires = {c1, target}});
  plan.append({.kind = PlanOpKind::P, .wires = {target}, .angle = quarter});
  plan.append({.kind = PlanOpKind::CX, .wires = {c0, target}});
  plan.append({.kind = PlanOpKind::P, .wires = {target}, .angle = -quarter});
  plan.append({.kind = PlanOpKind::CX, .wires = {c1, target}});
  plan.append({.kind = PlanOpKind::P, .wires = {target}, .angle = quarter});
  plan.append({.kind = PlanOpKind::CRZ, .wires = {c0, c1}, .angle = half});
  plan.append({.kind = PlanOpKind::P, .wires = {c0}, .angle = quarter});
  return plan;
}

/// General-angle MCP for `2 ≤ k < K_MCP_SP22_MIN_CONTROLS`: optimized C²P at
/// k=2, Vale at k=3, Vale + Barenco-relative residual at k=4.
static CircuitPlan planMcp(double theta, size_t numControls) {
  assert(numControls >= 2 && numControls < K_MCP_SP22_MIN_CONTROLS &&
         "planMcp covers only the pre-SP22 MCP band");
  if (numControls == 2) {
    return planMcpTwoControlled(theta);
  }
  if (numControls == K_MCP_VALE_RELATIVE_RESIDUAL_CONTROLS) {
    return planMcpValeRelativeResidual(theta, numControls);
  }
  return planMcpVale(theta, numControls);
}

//===----------------------------------------------------------------------===//
// SP22 linear-depth multi-controlled phase
//===----------------------------------------------------------------------===//

/// SP22 Eq. (1) `P_m` as single-controlled CRX ladder; `sign = -1` → dagger.
static void appendSp22PRx(CircuitPlan& plan, size_t m, double sign) {
  for (size_t c = 1; c < m; ++c) {
    plan.append({
        .kind = PlanOpKind::CRX,
        .wires = {c, m},
        .angle = sign * std::ldexp(K_PI, -static_cast<int>(m - c)),
    });
  }
}

/// SP22 Theorem 2: expand `Q_m` into CRX gates for `m >= 5`.
static CircuitPlan buildSp22Q(size_t m) {
  CircuitPlan q;
  q.ops.reserve((m - 1) * (m - 1));
  // Q_m = P_{m-1} CRX Q_{m-1} P_{m-1}^dagger. Emit the nested
  // prefixes first, then their suffixes, without moving child plans.
  for (size_t level = m; level > 1; --level) {
    appendSp22PRx(q, level - 1, 1.0);
    q.append({
        .kind = PlanOpKind::CRX,
        .wires = {0, level - 1},
        .angle = std::ldexp(K_PI, -static_cast<int>(level - 2)),
    });
  }
  for (size_t level = 2; level <= m; ++level) {
    appendSp22PRx(q, level - 1, -1.0);
  }
  return q;
}

/// SP22 LDD MCP (arXiv:2203.11882 Them. 1): CP ladder + CRX `Q_n` conjugation.
/// Controls `0..n-1`, target `n`; requires `n >= 5`.
static CircuitPlan planMcpSp22(double theta, size_t numControls) {
  CircuitPlan plan;
  const size_t n = numControls;
  const size_t target = n;
  plan.ops.reserve((2 * n * n) - (2 * n) + 1);

  const auto rootAngle = [&](double base, size_t exponent) {
    return std::ldexp(base, -static_cast<int>(exponent));
  };

  // P_n(U)
  for (size_t c = 1; c < n; ++c) {
    plan.append({
        .kind = PlanOpKind::CP,
        .wires = {c, target},
        .angle = rootAngle(theta, n - c),
    });
  }

  // Mid-root
  plan.append({
      .kind = PlanOpKind::CP,
      .wires = {0, target},
      .angle = rootAngle(theta, n - 1),
  });

  // Q_n
  const CircuitPlan qn = buildSp22Q(n);
  plan.ops.append(qn.ops.begin(), qn.ops.end());

  // P_n(U)^dagger
  for (size_t c = 1; c < n; ++c) {
    plan.append({
        .kind = PlanOpKind::CP,
        .wires = {c, target},
        .angle = rootAngle(-theta, n - c),
    });
  }

  // Q_n^dagger
  for (size_t i = qn.ops.size(); i-- > 0;) {
    const PlanOp& op = qn.ops[i];
    plan.append({.kind = op.kind, .wires = op.wires, .angle = -op.angle});
  }

  return plan;
}

// MCZ core: k=4 relative-phase C^4(Z); SP22 MCP(π) for 5..32; else HP24.
static CircuitPlan mczCoreForWidth(size_t numControls) {
  if (numControls == 4) {
    return planMczRelativePhaseK4();
  }
  if (numControls >= K_MCP_SP22_MIN_CONTROLS &&
      numControls <= K_MCX_SP22_MAX_CONTROLS) {
    return planMcpSp22(K_PI, numControls);
  }
  return planHp24Core(numControls);
}

// General-angle MCP: SP22 at k >= 5, else C²P / Vale (relative residual at 4).
static void emitMcpDefault(GateEmitter& emitter, double phi,
                           size_t numControls) {
  if (numControls >= K_MCP_SP22_MIN_CONTROLS) {
    lowerPlan(emitter, planMcpSp22(phi, numControls));
    return;
  }
  lowerPlan(emitter, planMcp(phi, numControls));
}

static SmallVector<Value> synthesizeMultiControlledPhase(OpBuilder& builder,
                                                         Location loc,
                                                         ValueRange controls,
                                                         Value target,
                                                         double theta) {
  SmallVector<Value> wires(controls.begin(), controls.end());
  wires.push_back(target);
  GateEmitter emitter(builder, loc, wires);
  emitMcpDefault(emitter, theta, controls.size());
  return wires;
}

//===----------------------------------------------------------------------===//
// CtrlOp body matchers
//===----------------------------------------------------------------------===//

/// Match a supported controlled-target body: Pauli-X, Pauli-Z, or a
/// constant-theta phase.
static std::optional<ControlledGateSpec>
matchControlledTarget(UnitaryOpInterface inner) {
  if (isa<XOp>(inner.getOperation())) {
    return ControlledGateSpec{
        .gate = ControlledTarget::X,
        .theta = std::nullopt,
    };
  }
  if (isa<ZOp>(inner.getOperation())) {
    return ControlledGateSpec{
        .gate = ControlledTarget::Z,
        .theta = std::nullopt,
    };
  }
  if (auto pOp = dyn_cast<POp>(inner.getOperation())) {
    if (const auto theta = mlir::mqt::valueToDouble(pOp.getTheta())) {
      return ControlledGateSpec{
          .gate = ControlledTarget::Phase,
          .theta = theta,
      };
    }
  }
  return std::nullopt;
}

/// Rewrite controlled-SWAP (Fredkin) as CX–MCX–CX.
///
/// Identity: MCSWAP(C, a, b) = CX(a, b) · MCX(C ∪ {b}, a) · CX(a, b).
/// Callers gate on the controlled-SWAP's total qubit count (`min-qubits`).
static SmallVector<Value>
synthesizeControlledSwap(OpBuilder& builder, Location loc, ValueRange controls,
                         Value targetA, Value targetB) {
  const auto makeX = [&](Value t) {
    return XOp::create(builder, loc, t).getOutputQubit(0);
  };

  auto cx1 = CtrlOp::create(builder, loc, targetA, targetB, makeX);

  SmallVector<Value, 4> mcxControls(controls);
  mcxControls.push_back(cx1.getOutputTarget(0));
  auto mcx =
      CtrlOp::create(builder, loc, mcxControls, cx1.getOutputControl(0), makeX);

  auto cx2 = CtrlOp::create(builder, loc, mcx.getOutputTarget(0),
                            mcx.getOutputControl(controls.size()), makeX);

  SmallVector<Value> results(mcx.getOutputControls().drop_back());
  results.push_back(cx2.getOutputControl(0));
  results.push_back(cx2.getOutputTarget(0));
  return results;
}

//===----------------------------------------------------------------------===//
// Patterns and pass
//===----------------------------------------------------------------------===//

static bool isWithinTargetNativeUnitary(UnitaryOpInterface op,
                                        const CompilerTarget* target) {
  if (target == nullptr) {
    return false;
  }
  for (; op; op = op->getParentOfType<UnitaryOpInterface>()) {
    if (target->supports(op.getOperation())) {
      return true;
    }
  }
  return false;
}

namespace {

struct DecomposeControlledGatePattern final : OpRewritePattern<CtrlOp> {
  explicit DecomposeControlledGatePattern(MLIRContext* context,
                                          uint64_t minQubits,
                                          const CompilerTarget* target)
      : OpRewritePattern<CtrlOp>(context), minQubits_(minQubits),
        target_(target) {}

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumQubits() < minQubits_) {
      return failure();
    }
    if (isWithinTargetNativeUnitary(op, target_)) {
      return failure();
    }

    const auto numControls = op.getNumControls();
    auto inner = mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }

    // MCSWAP(C, a, b) = CX(a,b) · MCX(C ∪ {b}, a) · CX(a,b).
    if (op.getNumTargets() == 2 && isa<SWAPOp>(inner.getOperation())) {
      rewriter.setInsertionPoint(op);
      rewriter.replaceOp(op, synthesizeControlledSwap(
                                 rewriter, op.getLoc(), op.getControlsIn(),
                                 op.getInputTarget(0), op.getInputTarget(1)));
      return success();
    }

    if (op.getNumTargets() != 1) {
      return failure();
    }
    if (isa<YOp>(inner.getOperation())) {
      // Y = S X S†; the new MCX reuses this pass's width selection.
      rewriter.setInsertionPoint(op);
      auto loc = op.getLoc();
      auto target =
          SdgOp::create(rewriter, loc, op.getInputTarget(0)).getOutputQubit(0);
      auto mcx = CtrlOp::create(
          rewriter, loc, op.getControlsIn(), target, [&](Value targetArg) {
            return XOp::create(rewriter, loc, targetArg).getOutputQubit(0);
          });
      SmallVector<Value> results(mcx.getOutputControls());
      results.push_back(
          SOp::create(rewriter, loc, mcx.getOutputTarget(0)).getOutputQubit(0));
      rewriter.replaceOp(op, results);
      return success();
    }
    if (isa<RXOp, RYOp, RZOp>(inner.getOperation())) {
      // Verified support operations cannot depend on the body's qubits.
      // Hoist them so region-local symbolic angles survive the replacement.
      mqt::hoistSupportingOpsBefore(*op.getBody(), inner.getOperation(), op,
                                    rewriter);
      rewriter.setInsertionPoint(op);
      rewriter.replaceOp(op, synthesizeMultiControlledRotation(
                                 rewriter, op.getLoc(), op.getControlsIn(),
                                 op.getInputTarget(0), inner));
      return success();
    }
    const auto spec = matchControlledTarget(inner);
    if (!spec) {
      return failure();
    }

    ControlledTarget gate = spec->gate;
    // A compile-time phase of +/- pi is exactly Z; route it through the
    // multi-controlled-Z path (elementary at 3–4 qubits, relative-phase at
    // 5 qubits, SP22 at 6–33 qubits, else HP24).
    if (gate == ControlledTarget::Phase && spec->theta &&
        std::abs(std::abs(*spec->theta) - K_PI) <=
            mqt::PARAMETER_COMPARISON_TOLERANCE) {
      gate = ControlledTarget::Z;
    }

    rewriter.setInsertionPoint(op);
    if (gate == ControlledTarget::Phase) {
      rewriter.replaceOp(op, synthesizeMultiControlledPhase(
                                 rewriter, op.getLoc(), op.getControlsIn(),
                                 op.getInputTarget(0), *spec->theta));
      return success();
    }
    if (numControls < 3) {
      // Exactly two controls (k < 2 is rejected by min-qubits >= 3).
      rewriter.replaceOp(op, synthesizeTwoControlled(
                                 rewriter, op.getLoc(), op.getControlsIn()[0],
                                 op.getControlsIn()[1], op.getInputTarget(0),
                                 gate));
      return success();
    }
    if (numControls == 3) {
      rewriter.replaceOp(op, synthesizeThreeControlled(
                                 rewriter, op.getLoc(), op.getControlsIn(),
                                 op.getInputTarget(0), gate));
    } else {
      rewriter.replaceOp(op, synthesizeMultiControlled(
                                 rewriter, op.getLoc(), op.getControlsIn(),
                                 op.getInputTarget(0), gate));
    }
    return success();
  }

private:
  uint64_t minQubits_;
  const CompilerTarget* target_;
};

struct DecomposeRCCXPattern final : OpRewritePattern<RCCXOp> {
  explicit DecomposeRCCXPattern(MLIRContext* context, uint64_t minQubits,
                                const CompilerTarget* target)
      : OpRewritePattern<RCCXOp>(context), minQubits_(minQubits),
        target_(target) {}

  LogicalResult matchAndRewrite(RCCXOp op,
                                PatternRewriter& rewriter) const override {
    if (RCCXOp::getNumQubits() < minQubits_) {
      return failure();
    }
    if (isWithinTargetNativeUnitary(op, target_)) {
      return failure();
    }
    rewriter.setInsertionPoint(op);
    rewriter.replaceOp(
        op, synthesizeRCCX(rewriter, op.getLoc(), op.getInputQubit(0),
                           op.getInputQubit(1), op.getInputQubit(2)));
    return success();
  }

private:
  uint64_t minQubits_;
  const CompilerTarget* target_;
};

struct DecomposeMultiControlled final
    : impl::DecomposeMultiControlledBase<DecomposeMultiControlled> {
  using DecomposeMultiControlledBase::DecomposeMultiControlledBase;

  DecomposeMultiControlled(const CompilerTarget& target, uint64_t minQubitsIn)
      : target_(target) {
    minQubits = minQubitsIn;
  }

protected:
  void runOnOperation() override {
    if (minQubits < 3) {
      getOperation().emitError()
          << "decompose-multi-controlled requires min-qubits >= 3";
      signalPassFailure();
      return;
    }

    const CompilerTarget* nativeTarget =
        target_ && target_->connectivityKind() ==
                       CompilerTarget::Connectivity::Kind::AllToAll
            ? &*target_
            : nullptr;

    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeControlledGatePattern, DecomposeRCCXPattern>(
        &getContext(), minQubits, nativeTarget);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

private:
  std::optional<CompilerTarget> target_;
};

} // namespace

std::unique_ptr<Pass>
createDecomposeMultiControlled(const CompilerTarget& target,
                               uint64_t minQubits) {
  return std::make_unique<DecomposeMultiControlled>(target, minQubits);
}

} // namespace mlir::qco
