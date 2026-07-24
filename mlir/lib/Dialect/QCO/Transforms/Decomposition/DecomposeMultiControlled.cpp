/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_DECOMPOSEMULTICONTROLLED
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

enum class Hp24DirtyMode : uint8_t { OneDirty, TwoDirty };
enum class Hp24IncrementerKind : uint8_t { Ripple, Partitioned };
enum class Hp24HalfMcxKind : uint8_t { RelativePhaseTernary, BorrowedHelper };
struct Hp24Policy {
  Hp24DirtyMode dirtyMode = Hp24DirtyMode::TwoDirty;
  size_t halfSplit = 0;
  Hp24IncrementerKind incrementerKind = Hp24IncrementerKind::Ripple;
  size_t incrementerRippleMaxWidth = 10;
  Hp24HalfMcxKind halfMcxKind = Hp24HalfMcxKind::RelativePhaseTernary;
  size_t halfMcxBorrowedHelperMinControls = 11;
};

enum class ControlledTarget : uint8_t { X, Z, Phase };

static constexpr double K_PI = std::numbers::pi;
static constexpr double K_PI8 = K_PI / 8.0;

class GateEmitter {
public:
  GateEmitter(OpBuilder& builder, Location loc, SmallVector<Value>& wires,
              ArrayRef<size_t> remap = {})
      : builder_(&builder), loc_(loc), wires_(&wires), remap_(remap) {}

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

  // Building blocks left as QCO ops (further lowered by min-controls)
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

  // Arbitrary-width multi-controlled X, left as a `qco.ctrl` op for further
  // decomposition (e.g. by this pass's own greedy rewriting, or by a nested
  // plan). Used for the `NestedMCX` `PlanOp` kind.
  void emitCtrlX(ArrayRef<size_t> controls, size_t target) {
    emitCtrl(controls, target, [](OpBuilder& builder, Location loc, Value arg) {
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

  [[nodiscard]] size_t wireIndex(size_t local) const {
    return remap_.empty() ? local : remap_[local];
  }

  [[nodiscard]] Value wire(size_t local) const {
    return (*wires_)[wireIndex(local)];
  }

  void setWire(size_t local, Value value) {
    (*wires_)[wireIndex(local)] = value;
  }

  OpBuilder* builder_;
  Location loc_;
  SmallVector<Value>* wires_;
  ArrayRef<size_t> remap_;
};

//===----------------------------------------------------------------------===//
// Circuit plan
//===----------------------------------------------------------------------===//

/// Plan-level op kinds. `NestedMCX` is an arbitrary-width multi-controlled X.
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
  NestedMCX
};

/// One plan op. `wires` are local indices for `lowerPlan`; for `NestedMCX`,
/// controls are `wires[0 .. nestedControls)` and the target is
/// `wires[nestedControls]`.
struct PlanOp {
  PlanOpKind kind{};
  SmallVector<size_t, 4> wires;
  double angle = 0.0;
  size_t nestedControls = 0;
};

/// Ordered plan ops lowered by `lowerPlan`.
struct CircuitPlan {
  SmallVector<PlanOp, 32> ops;

  void append(PlanOp op) { ops.push_back(std::move(op)); }
};

[[nodiscard]] static size_t estimateBorrowedHelperMcxOps(size_t numControls) {
  if (numControls <= 3) {
    return 1;
  }
  // Two passes: CCX + (n-3)*5 gadget + RCCX + (n-3)*5 gadget each.
  return 2 * (2 + (10 * (numControls - 3)));
}

[[nodiscard]] static size_t estimateRelativePhaseMcxOps(size_t numControls) {
  if (numControls <= 2) {
    return 1;
  }
  const size_t num3 = numControls / 3;
  const size_t num2 = (numControls - num3) / 2;
  const size_t num1 = numControls - num3 - num2;
  return 9 + (4 * estimateRelativePhaseMcxOps(num3)) +
         (2 * estimateRelativePhaseMcxOps(num2)) +
         (2 * estimateRelativePhaseMcxOps(num1));
}

[[nodiscard]] static size_t estimateIncrementerPartitionedOps(size_t n) {
  return (16 * n) + 4;
}

[[nodiscard]] static size_t estimateIncrementerRippleOps(size_t n) {
  size_t total = 1;
  for (size_t width = 1; width < n; ++width) {
    total += estimateBorrowedHelperMcxOps(width);
  }
  return total;
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
    case PlanOpKind::NestedMCX: {
      const ArrayRef<size_t> controls =
          ArrayRef<size_t>(op.wires).take_front(op.nestedControls);
      emitter.emitCtrlX(controls, op.wires[op.nestedControls]);
      break;
    }
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

struct ControlledGateSpec {
  ControlledTarget gate;
  std::optional<double> theta;
};

//===----------------------------------------------------------------------===//
// HP24 MCZ core (Huang & Palsberg, PACMPL 2024, doi:10.1145/3656436)
//===----------------------------------------------------------------------===//
// Phase-π core on all-ones; no clean helpers (borrow target / a control as
// dirty). Callers use `MCZ = core` and `MCX = H . core . H` on the target.

static constexpr size_t K_ONE_DIRTY_MIN_CONTROLS = 23;
static constexpr size_t K_HP24_POLICY_TABLE_MIN = 4;
static constexpr size_t K_HP24_POLICY_TABLE_MAX = 24;

[[nodiscard]] static constexpr Hp24Policy
defaultHp24Policy(size_t numControls) {
  Hp24Policy policy;
  if (numControls >= K_ONE_DIRTY_MIN_CONTROLS && (numControls % 2 == 1)) {
    policy.dirtyMode = Hp24DirtyMode::OneDirty;
  }
  return policy;
}

// HP24 policies for k=4…24 (`selectHp24Policy`).
static constexpr auto K_HP24_POLICY_TABLE = [] {
  std::array<Hp24Policy, K_HP24_POLICY_TABLE_MAX + 1> table{};
  for (size_t k = 0; k <= K_HP24_POLICY_TABLE_MAX; ++k) {
    table[k] = defaultHp24Policy(k);
  }
  table[5].dirtyMode = Hp24DirtyMode::OneDirty;
  table[21].halfMcxBorrowedHelperMinControls = 13;
  table[22].halfMcxBorrowedHelperMinControls = 13;
  return table;
}();

[[nodiscard]] static Hp24Policy selectHp24Policy(size_t numControls) {
  if (numControls >= K_HP24_POLICY_TABLE_MIN &&
      numControls <= K_HP24_POLICY_TABLE_MAX) {
    return K_HP24_POLICY_TABLE[numControls];
  }
  return defaultHp24Policy(numControls);
}

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

// HP24 Fig. 5 carry / uncarry steps for the borrowed-helper incrementer.
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

// HP24 Fig. 6 borrowed-helper incrementer `U^n_{+1}` (wide registers).
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

// HP24 Fig. 10 ripple incrementer (narrow registers).
static CircuitPlan planIncrementerRipple(size_t n) {
  CircuitPlan plan;
  plan.ops.reserve(estimateIncrementerRippleOps(n));
  SmallVector<size_t, 16> wires;
  for (size_t width = n - 1; width >= 1; --width) {
    wires.clear();
    for (size_t q = 0; q <= width; ++q) {
      wires.push_back(q);
    }
    for (size_t q = n + 1; q < 2 * n; ++q) {
      wires.push_back(q);
    }
    appendRemapped(plan, planBorrowedHelperMcx(width), wires);
  }
  plan.append({.kind = PlanOpKind::X, .wires = {0}});
  return plan;
}

// `U^n_{+1}`: Fig. 6 when wide, Fig. 10 ripple when narrow (crossover n=10).
static CircuitPlan planIncrementer(size_t n, const Hp24Policy& policy) {
  if (policy.incrementerKind == Hp24IncrementerKind::Ripple &&
      n <= policy.incrementerRippleMaxWidth) {
    return planIncrementerRipple(n);
  }
  return planIncrementerPartitioned(n);
}

// HP24 §4.3 relative-phase MCX (ternary ladder); phases cancel in pairs.
static CircuitPlan planRelativePhaseMcx(size_t numControls) {
  CircuitPlan plan;
  const size_t target = numControls;
  if (numControls == 0) {
    return plan;
  }
  if (numControls == 1) {
    plan.append({.kind = PlanOpKind::CX, .wires = {0, 1}});
    return plan;
  }
  if (numControls == 2) {
    plan.append({.kind = PlanOpKind::RCCX, .wires = {0, 1, 2}});
    return plan;
  }

  plan.ops.reserve(estimateRelativePhaseMcxOps(numControls));

  // Balanced three-way split of the controls into blocks of sizes num1, num2,
  // num3 (num3 = floor(k/3) is the largest split that keeps the ladder
  // balanced across the recursion).
  const size_t num3 = numControls / 3;
  const size_t num2 = (numControls - num3) / 2;
  const size_t num1 = numControls - num3 - num2;
  const size_t block2Begin = num1;
  const size_t block3Begin = num1 + num2;
  const size_t controlsEnd = numControls;

  SmallVector<size_t, 16> wires;
  const auto ladderStep = [&](size_t begin, size_t end, size_t width,
                              bool positive) {
    plan.append({.kind = PlanOpKind::P,
                 .wires = {target},
                 .angle = positive ? K_PI8 : -K_PI8});
    wires.clear();
    for (size_t q = begin; q < end; ++q) {
      wires.push_back(q);
    }
    wires.push_back(target);
    appendRemapped(plan, planRelativePhaseMcx(width), wires);
  };

  plan.append({.kind = PlanOpKind::H, .wires = {target}});
  ladderStep(block3Begin, controlsEnd, num3, true);
  ladderStep(block2Begin, block3Begin, num2, false);
  ladderStep(block3Begin, controlsEnd, num3, true);
  ladderStep(0, block2Begin, num1, false);
  ladderStep(block3Begin, controlsEnd, num3, true);
  ladderStep(block2Begin, block3Begin, num2, false);
  ladderStep(block3Begin, controlsEnd, num3, true);
  ladderStep(0, block2Begin, num1, false);
  plan.append({.kind = PlanOpKind::H, .wires = {target}});
  return plan;
}

// Ternary relative-phase MCX below helperMin; else borrowed-helper MCX.
static CircuitPlan planRelativePhaseMcxWide(size_t numControls,
                                            const Hp24Policy& policy) {
  if (policy.halfMcxKind == Hp24HalfMcxKind::RelativePhaseTernary &&
      numControls < policy.halfMcxBorrowedHelperMinControls) {
    return planRelativePhaseMcx(numControls);
  }
  return planBorrowedHelperMcx(numControls);
}

// HP24 Fig. 6/8 partitioned incrementer. One-dirty borrows the target;
// two-dirty also borrows the top control. `flagAdd == false` yields `U_{-1}`
// (Eq. (7)).
static CircuitPlan planBorrowedDirtyIncrementer(size_t n, bool flagAdd,
                                                const Hp24Policy& policy) {
  CircuitPlan plan;
  const bool oneDirty = policy.dirtyMode == Hp24DirtyMode::OneDirty;
  const size_t numDirty = oneDirty ? 1 : 2;
  size_t k = policy.halfSplit;
  if (k == 0) {
    k = oneDirty ? (n + 1) / 2 : (n + 2) / 2;
  }
  const size_t helper = n;
  const size_t helper2 = n + 1;
  const size_t lowIncrementWidth = oneDirty ? k : (1 + n - k);
  const size_t incrementerOps =
      policy.incrementerKind == Hp24IncrementerKind::Ripple &&
              lowIncrementWidth <= policy.incrementerRippleMaxWidth
          ? estimateIncrementerRippleOps(lowIncrementWidth)
          : estimateIncrementerPartitionedOps(lowIncrementWidth);
  const size_t halfMcxOps =
      policy.halfMcxKind == Hp24HalfMcxKind::RelativePhaseTernary &&
              k < policy.halfMcxBorrowedHelperMinControls
          ? estimateRelativePhaseMcxOps(k)
          : estimateBorrowedHelperMcxOps(k);
  const size_t highIncrementOps =
      policy.incrementerKind == Hp24IncrementerKind::Ripple &&
              k <= policy.incrementerRippleMaxWidth
          ? estimateIncrementerRippleOps(k)
          : estimateIncrementerPartitionedOps(k);
  plan.ops.reserve((2 * incrementerOps) + (2 * halfMcxOps) + highIncrementOps +
                   (2 * (n - k)) + 4 + (flagAdd ? 0 : (2 * n)));

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

  // Final sub-incrementer over the high half: wire order [low half, high half,
  // helper, (helper2)].
  SmallVector<size_t, 16> highIncrementWires;
  for (size_t q = 0; q < k; ++q) {
    highIncrementWires.push_back(q);
  }
  for (size_t q = k; q < n; ++q) {
    highIncrementWires.push_back(q);
  }
  highIncrementWires.push_back(helper);
  if (numDirty == 2) {
    highIncrementWires.push_back(helper2);
  }

  const auto incrementLow = [&] {
    appendRemapped(plan, planIncrementer(lowIncrementWidth, policy),
                   lowIncrementWires);
  };
  const auto halfMcx = [&] {
    // Relative-phase / borrowed-helper MCX: the high half (and optional
    // helper2) on `halfMcxWires` are dirty workspace and must stay in the
    // remap map — a bare NestedMCX would drop them.
    appendRemapped(plan, planRelativePhaseMcxWide(k, policy), halfMcxWires);
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
  appendRemapped(plan, planIncrementer(k, policy), highIncrementWires);

  if (!flagAdd) {
    flipRegister();
  }
  return plan;
}

// HP24 Theorem 4.4: `C^{n-1}(p(π))` via dirty incrementer + phase ladder.
static CircuitPlan planHp24Core(size_t n, const Hp24Policy& policy) {
  CircuitPlan plan;
  const size_t numControls = n - 1;
  const size_t target = n - 1;
  const size_t topControl = n - 2;
  const size_t registerWidth = policy.dirtyMode == Hp24DirtyMode::OneDirty
                                   ? numControls
                                   : numControls - 1;
  const size_t incrementerOps =
      policy.incrementerKind == Hp24IncrementerKind::Ripple &&
              registerWidth <= policy.incrementerRippleMaxWidth
          ? estimateIncrementerRippleOps(registerWidth)
          : estimateIncrementerPartitionedOps(registerWidth);
  plan.ops.reserve((2 * incrementerOps) + (2 * (numControls - 1)) + 1);

  SmallVector<size_t, 16> registerWires(n);
  std::iota(registerWires.begin(), registerWires.end(), 0U);

  if (policy.dirtyMode == Hp24DirtyMode::OneDirty) {
    const auto increment = [&](bool add) {
      appendRemapped(plan,
                     planBorrowedDirtyIncrementer(numControls, add, policy),
                     registerWires);
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
    appendRemapped(plan,
                   planBorrowedDirtyIncrementer(numControls - 1, add, policy),
                   registerWires);
  };
  increment(true);
  double phi = -K_PI;
  for (size_t q = numControls - 2; q > 0; --q) {
    phi /= 2.0;
    plan.append({.kind = PlanOpKind::CCP,
                 .wires = {q, topControl, target},
                 .angle = phi});
  }
  increment(false);
  phi = K_PI;
  for (size_t q = numControls - 2; q > 0; --q) {
    phi /= 2.0;
    plan.append({.kind = PlanOpKind::CCP,
                 .wires = {q, topControl, target},
                 .angle = phi});
  }
  plan.append({.kind = PlanOpKind::CCP,
               .wires = {0, topControl, target},
               .angle = phi});
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

static SmallVector<Value>
synthesizeTwoControlled(OpBuilder& builder, Location loc, Value control0,
                        Value control1, Value target, ControlledTarget gate,
                        std::optional<double> /*theta*/ = std::nullopt) {
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
    // Two-controlled phase is handled by synthesizeMultiControlledPhase /
    // planMcpTwoControlled (also the Vale residual at k=3).
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

// Barenco peel with RCCX at width 2; exact NestedMCX for wider peels.
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
    PlanOp mcx{.kind = PlanOpKind::NestedMCX, .nestedControls = peeled};
    mcx.wires.reserve(peeled + 1);
    for (size_t control = 0; control < peeled; ++control) {
      mcx.wires.push_back(control);
    }
    mcx.wires.push_back(peeled);
    plan.append(std::move(mcx));
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
  const std::array<PlanOp, 18> ops = {{
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
  }};
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

static CircuitPlan mczCoreForWidth(size_t numControls, size_t numWires);

static SmallVector<Value>
synthesizeMultiControlled(OpBuilder& builder, Location loc, ValueRange controls,
                          Value target, ControlledTarget gate) {
  SmallVector<Value> wires(controls.begin(), controls.end());
  wires.push_back(target);

  const size_t targetIdx = controls.size();
  GateEmitter emitter(builder, loc, wires);
  const CircuitPlan plan = mczCoreForWidth(controls.size(), wires.size());
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

static CircuitPlan planMcp(double theta, size_t numControls);

//===----------------------------------------------------------------------===//
// Vale multi-controlled phase
//===----------------------------------------------------------------------===//

/// Vale control split: top `ceil(k/2)`, bottom `floor(k/2)`.
struct BorrowedControlPartition {
  size_t k1; // ceil(k / 2)
  size_t k2; // floor(k / 2)
};

static BorrowedControlPartition partitionControls(size_t numControls) {
  return {.k1 = (numControls + 1) / 2, .k2 = numControls / 2};
}

// Vale + Barenco-relative residual at this MCP width.
static constexpr size_t K_MCP_VALE_RELATIVE_RESIDUAL_CONTROLS = 4;

/// Vale24 Fig. 7 shell (arXiv:2302.06377): alternate half-MCX with target
/// `p(±θ/4)`. Controls then target. Caller appends the residual.
static void appendValeFig7Shell(CircuitPlan& plan, double theta,
                                size_t numControls) {
  const size_t target = numControls;
  const auto [k1, k2] = partitionControls(numControls);
  const double quarter = theta / 4.0;
  const auto appendHalfMcx = [&](size_t begin, size_t count) {
    PlanOp mcx{.kind = PlanOpKind::NestedMCX, .nestedControls = count};
    mcx.wires.reserve(count + 1);
    for (size_t c = 0; c < count; ++c) {
      mcx.wires.push_back(begin + c);
    }
    mcx.wires.push_back(target);
    plan.append(std::move(mcx));
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

/// Recursive Vale shells; Barenco-relative residual at width ≤ 4.
/// Used for MCX/MCZ k=5 (shell at 5, then relative residual at 4).
static CircuitPlan planMcpValeHybridResidual(double theta, size_t numControls) {
  if (numControls <= K_MCP_VALE_RELATIVE_RESIDUAL_CONTROLS) {
    return planMcpValeRelativeResidual(theta, numControls);
  }
  CircuitPlan plan;
  appendValeFig7Shell(plan, theta, numControls);
  appendPlanOps(plan, planMcpValeHybridResidual(theta / 2.0, numControls - 1));
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
    plan.append({.kind = PlanOpKind::CRX,
                 .wires = {c, m},
                 .angle = sign * std::ldexp(K_PI, -static_cast<int>(m - c))});
  }
}

/// SP22 Theorem 2: expand `Q_m` into single-controlled CRX only.
static CircuitPlan buildSp22Q(size_t m) {
  CircuitPlan q;
  if (m < 2) {
    return q; // Q_1 = Q_0 = I
  }
  appendSp22PRx(q, m - 1, 1.0);
  q.append({.kind = PlanOpKind::CRX,
            .wires = {0, m - 1},
            .angle = std::ldexp(K_PI, -static_cast<int>(m - 2))});
  appendPlanOps(q, buildSp22Q(m - 1));
  appendSp22PRx(q, m - 1, -1.0);
  return q;
}

/// SP22 LDD MCP (arXiv:2203.11882 Them. 1): CP ladder + CRX `Q_n` conjugation.
/// Controls `0..n-1`, target `n`.
static CircuitPlan planMcpSp22(double theta, size_t numControls) {
  CircuitPlan plan;
  const size_t n = numControls;
  const size_t target = n;
  if (n < 2) {
    // Should not be reached; k = 1 is an elementary CP handled elsewhere.
    plan.append({.kind = PlanOpKind::CP, .wires = {0, target}, .angle = theta});
    return plan;
  }
  plan.ops.reserve((2 * n * n) - (2 * n) + 1);

  const auto rootAngle = [&](double base, size_t exponent) {
    return std::ldexp(base, -static_cast<int>(exponent));
  };

  // P_n(U)
  for (size_t c = 1; c < n; ++c) {
    plan.append({.kind = PlanOpKind::CP,
                 .wires = {c, target},
                 .angle = rootAngle(theta, n - c)});
  }

  // Mid-root
  plan.append({.kind = PlanOpKind::CP,
               .wires = {0, target},
               .angle = rootAngle(theta, n - 1)});

  // Q_n
  const CircuitPlan qn = buildSp22Q(n);
  appendPlanOps(plan, qn);

  // P_n(U)^dagger
  for (size_t c = 1; c < n; ++c) {
    plan.append({.kind = PlanOpKind::CP,
                 .wires = {c, target},
                 .angle = rootAngle(-theta, n - c)});
  }

  // Q_n^dagger
  for (size_t i = qn.ops.size(); i-- > 0;) {
    const PlanOp& op = qn.ops[i];
    plan.append({.kind = op.kind, .wires = op.wires, .angle = -op.angle});
  }

  return plan;
}

// MCZ core: k=4 relative-phase C^4(Z); k=5 Vale hybrid; else HP24.
static CircuitPlan mczCoreForWidth(size_t numControls, size_t numWires) {
  if (numControls == 4) {
    return planMczRelativePhaseK4();
  }
  if (numControls == 5) {
    return planMcpValeHybridResidual(K_PI, numControls);
  }
  return planHp24Core(numWires, selectHp24Policy(numControls));
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
    return ControlledGateSpec{.gate = ControlledTarget::X,
                              .theta = std::nullopt};
  }
  if (isa<ZOp>(inner.getOperation())) {
    return ControlledGateSpec{.gate = ControlledTarget::Z,
                              .theta = std::nullopt};
  }
  if (auto pOp = dyn_cast<POp>(inner.getOperation())) {
    if (const auto theta = utils::valueToDouble(pOp.getTheta())) {
      return ControlledGateSpec{.gate = ControlledTarget::Phase,
                                .theta = theta};
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Patterns and pass
//===----------------------------------------------------------------------===//

struct DecomposeControlledGatePattern final : OpRewritePattern<CtrlOp> {
  explicit DecomposeControlledGatePattern(MLIRContext* context,
                                          uint64_t minControls)
      : OpRewritePattern<CtrlOp>(context), minControls_(minControls) {}

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    const auto numControls = op.getNumControls();
    if (numControls < minControls_ || op.getNumTargets() != 1) {
      return failure();
    }
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    const auto spec = matchControlledTarget(inner);
    if (!spec) {
      return failure();
    }

    rewriter.setInsertionPoint(op);
    if (numControls < 3) {
      // Exactly two controls (k < 2 is rejected by min-controls >= 2).
      // Route two-controlled phase through the shared MCP planner (optimized
      // C²P), which is also the Vale residual at k=3.
      if (spec->gate == ControlledTarget::Phase) {
        rewriter.replaceOp(op, synthesizeMultiControlledPhase(
                                   rewriter, op.getLoc(), op.getControlsIn(),
                                   op.getInputTarget(0), *spec->theta));
        return success();
      }
      rewriter.replaceOp(op, synthesizeTwoControlled(
                                 rewriter, op.getLoc(), op.getControlsIn()[0],
                                 op.getControlsIn()[1], op.getInputTarget(0),
                                 spec->gate, spec->theta));
      return success();
    }

    ControlledTarget gate = spec->gate;
    // A compile-time phase of +/- pi is exactly Z; route it through the
    // multi-controlled-Z path (Vale MCP(π)+Barenco-relative at k=4/5, else
    // HP24).
    if (gate == ControlledTarget::Phase && spec->theta &&
        std::abs(std::abs(*spec->theta) - K_PI) <= utils::TOLERANCE) {
      gate = ControlledTarget::Z;
    }
    if (gate == ControlledTarget::Phase) {
      rewriter.replaceOp(op, synthesizeMultiControlledPhase(
                                 rewriter, op.getLoc(), op.getControlsIn(),
                                 op.getInputTarget(0), *spec->theta));
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
  uint64_t minControls_;
};

struct DecomposeRCCXPattern final : OpRewritePattern<RCCXOp> {
  explicit DecomposeRCCXPattern(MLIRContext* context, uint64_t minControls)
      : OpRewritePattern<RCCXOp>(context), minControls_(minControls) {}

  LogicalResult matchAndRewrite(RCCXOp op,
                                PatternRewriter& rewriter) const override {
    if (minControls_ > 2) {
      return failure();
    }
    rewriter.setInsertionPoint(op);
    rewriter.replaceOp(
        op, synthesizeRCCX(rewriter, op.getLoc(), op.getInputQubit(0),
                           op.getInputQubit(1), op.getInputQubit(2)));
    return success();
  }

private:
  uint64_t minControls_;
};

struct DecomposeMultiControlled final
    : impl::DecomposeMultiControlledBase<DecomposeMultiControlled> {
  using DecomposeMultiControlledBase::DecomposeMultiControlledBase;

protected:
  void runOnOperation() override {
    if (minControls < 2) {
      getOperation().emitError()
          << "decompose-multi-controlled requires min-controls >= 2";
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeControlledGatePattern, DecomposeRCCXPattern>(
        &getContext(), minControls);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::qco
