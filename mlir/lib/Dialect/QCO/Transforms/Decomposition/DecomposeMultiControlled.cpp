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
#include "mqt/Dialect/MQT/Transforms/UnrollModifiers.h"
#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/Modifiers.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h" // IWYU pragma: keep (Passes.h.inc)
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numbers>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>

namespace mlir::qco {

#define GEN_PASS_DEF_DECOMPOSEMULTICONTROLLED
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

// The synthesis routines below are independent implementations derived from
// the publications cited at the respective algorithms.

enum class Hp24DirtyMode : uint8_t { OneDirty, TwoDirty };
enum class ControlledTarget : uint8_t { X, Z };

constexpr double K_PI = std::numbers::pi;
constexpr double K_PI8 = K_PI / 8.0;

class GateEmitter {
public:
  GateEmitter(OpBuilder& builder, Location loc, MutableArrayRef<Value> wires)
      : builder_(&builder), loc_(loc), wires_(wires) {}

  // Single- and two-qubit primitives
  void h(size_t q) {
    setWire(q, HOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void x(size_t q) {
    setWire(q, XOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void p(size_t q, const mqt::FloatParameter& theta) {
    setWire(q, POp::create(*builder_, loc_, wire(q), theta).getOutputQubit(0));
  }

  void rz(size_t q, const mqt::FloatParameter& theta) {
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

  void cp(size_t control, size_t target, const mqt::FloatParameter& theta) {
    auto ctrlOp =
        CtrlOp::create(*builder_, loc_, wire(control), wire(target),
                       [&](Value targetArg) -> Value {
                         return POp::create(*builder_, loc_, targetArg, theta)
                             .getOutputQubit(0);
                       });
    setWire(control, ctrlOp.getControlsOut()[0]);
    setWire(target, ctrlOp.getTargetsOut()[0]);
  }

  void crz(size_t control, size_t target, const mqt::FloatParameter& theta) {
    const auto half = mqt::scaleParameter(*builder_, loc_, theta, 0.5);
    p(target, half);
    cx(control, target);
    p(target, mqt::scaleParameter(*builder_, loc_, half, -1.0));
    cx(control, target);
  }

  // Controlled-RX via RX(θ) = H RZ(θ) H, reusing crz.
  void crx(size_t control, size_t target, const mqt::FloatParameter& theta) {
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

  void ccp(const mqt::FloatParameter& theta, size_t c0, size_t c1,
           size_t target) {
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

  [[nodiscard]] Value wire(size_t local) const { return wires_[local]; }

  void setWire(size_t local, Value value) { wires_[local] = value; }

  OpBuilder* builder_;
  Location loc_;
  MutableArrayRef<Value> wires_;
};

//===----------------------------------------------------------------------===//
// Circuit plan
//===----------------------------------------------------------------------===//

/// Plan-level op kinds.
enum class PlanOpKind : uint8_t {
  H,
  X,
  P,
  RZ,
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
  mqt::FloatParameter angle = 0.0;
};

/// Ordered plan ops lowered by `lowerPlan`.
struct CircuitPlan {
  SmallVector<PlanOp> ops;

  void append(PlanOp op) { ops.push_back(std::move(op)); }
};

/// A helper is a live wire, not spare target-device capacity. Tensor slots
/// must be provably occupied; extracting a hole would duplicate a qubit.
struct BorrowedHelper {
  Value value;
  std::optional<int64_t> slot;
  bool clean;
};

} // namespace

static bool isPiPhase(const mqt::FloatParameter& angle) {
  const auto theta = mqt::parameterToConstantDouble(angle);
  return theta && std::abs(std::abs(*theta) - K_PI) <=
                      mqt::PARAMETER_COMPARISON_TOLERANCE;
}

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
estimateBorrowedDirtyIncrementerOps(size_t n, Hp24DirtyMode dirtyMode) {
  const bool oneDirty = dirtyMode == Hp24DirtyMode::OneDirty;
  const size_t k = oneDirty ? (n + 1) / 2 : (n + 2) / 2;
  const size_t lowIncrementWidth = oneDirty ? k : (1 + n - k);
  const size_t incrementerOps =
      estimateIncrementerPartitionedOps(lowIncrementWidth);
  const size_t halfMcxOps = estimateBorrowedHelperMcxOps(k);
  const size_t highIncrementOps = estimateIncrementerPartitionedOps(k);
  return (2 * incrementerOps) + (2 * halfMcxOps) + highIncrementOps +
         (2 * (n - k)) + 4;
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
    case PlanOpKind::RZ:
      emitter.rz(op.wires[0], op.angle);
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
  const double quarterPi = K_PI / 4.0; // T = p(π/4), Tdg = p(-π/4)
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

// Relative-phase HP24 Fig. 5 carries: same basis permutation, cheaper phases.
static void appendRelativeCarry(CircuitPlan& plan, size_t carry, size_t hi,
                                size_t lo) {
  plan.append({.kind = PlanOpKind::CX, .wires = {carry, lo}});
  plan.append({.kind = PlanOpKind::CX, .wires = {carry, hi}});
  plan.append({.kind = PlanOpKind::RCCX, .wires = {hi, lo, carry}});
}

static void appendRelativeUncarry(CircuitPlan& plan, size_t carry, size_t hi,
                                  size_t lo) {
  plan.append({.kind = PlanOpKind::RCCX, .wires = {hi, lo, carry}});
  plan.append({.kind = PlanOpKind::CX, .wires = {carry, hi}});
  plan.append({.kind = PlanOpKind::CX, .wires = {hi, lo}});
}

/// Hermitian carry block shared by the borrowed-MCX and rotation plans.
static CircuitPlan planBorrowedMcxCarry(size_t k) {
  CircuitPlan carry;
  for (size_t i = k - 3; i-- > 0;) {
    appendGadget(carry, i + 2, k + 1 + i, k + 2 + i, false);
  }
  carry.append({.kind = PlanOpKind::RCCX, .wires = {0, 1, k + 1}});
  for (size_t i = 0; i < k - 3; ++i) {
    appendGadget(carry, i + 2, k + 1 + i, k + 2 + i, true);
  }
  return carry;
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
  const size_t lastHelper = 2 * numControls - 2;
  const CircuitPlan carry = planBorrowedMcxCarry(numControls);

  for (size_t pass = 0; pass < 2; ++pass) {
    plan.append(
        {.kind = PlanOpKind::CCX, .wires = {topControl, lastHelper, target}});
    appendPlanOps(plan, carry);
  }
  return plan;
}

// Relative-phase incrementer: restore helpers' bits, not their phases yet.
static CircuitPlan planRelativeIncrementerPartitioned(size_t n) {
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
      appendRelativeCarry(plan, carry, carry + 1 + i, i);
    }
  };
  const auto sweepDown = [&] {
    for (size_t i = n - 1; i-- > 0;) {
      appendRelativeUncarry(plan, carry, carry + 1 + i, i);
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

// Relative-phase HP24 Fig. 6/8 incrementer. One-dirty borrows the target;
// two-dirty also borrows the top control. Use only with its exact adjoint
// around a diagonal phase ladder, which cancels all these relative phases.
static CircuitPlan planRelativeBorrowedIncrementer(size_t n,
                                                   Hp24DirtyMode dirtyMode) {
  CircuitPlan plan;
  const bool oneDirty = dirtyMode == Hp24DirtyMode::OneDirty;
  const size_t numDirty = oneDirty ? 1 : 2;
  const size_t k = oneDirty ? (n + 1) / 2 : (n + 2) / 2;
  const size_t helper = n;
  const size_t helper2 = n + 1;
  const size_t lowIncrementWidth = oneDirty ? k : (1 + n - k);
  plan.ops.reserve(estimateBorrowedDirtyIncrementerOps(n, dirtyMode));

  // Sub-incrementer over the low half: wire order [helper, high half, low half,
  // (helper2)]; the trailing helpers become the borrowed workspace of `U_{+1}`.
  SmallVector<size_t> lowIncrementWires;
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
  SmallVector<size_t> halfMcxWires;
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
    appendRemapped(plan, planRelativeIncrementerPartitioned(lowIncrementWidth),
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

  incrementLow();
  plan.append({.kind = PlanOpKind::X, .wires = {helper}});
  fanOutHelper();
  halfMcx();
  incrementLow();
  plan.append({.kind = PlanOpKind::X, .wires = {helper}});
  halfMcx();
  fanOutHelper();
  appendPlanOps(plan, planRelativeIncrementerPartitioned(k));
  return plan;
}

// HP24 Theorem 4.4 with relative-phase incrementers. If V = D U, where U
// is the exact increment and D is diagonal on all wires (including helpers),
// then V† L V = U† L U for a diagonal ladder L. Use the literal adjoint:
// complementing an incrementer's input/output bits would not undo its phases.
static CircuitPlan planHp24Core(size_t numControls) {
  assert(numControls >= 9 && "narrower widths use the phase-corrected RZ plan");
  const size_t n = numControls + 1;
  const auto dirtyMode =
      numControls % 2 == 1 ? Hp24DirtyMode::OneDirty : Hp24DirtyMode::TwoDirty;
  CircuitPlan plan;
  const size_t target = n - 1;
  const size_t topControl = n - 2;
  const size_t registerWidth =
      dirtyMode == Hp24DirtyMode::OneDirty ? numControls : numControls - 1;
  const CircuitPlan compute =
      planRelativeBorrowedIncrementer(registerWidth, dirtyMode);
  plan.ops.reserve(2 * compute.ops.size() + 2 * numControls - 1);
  const auto increment = [&](bool add) {
    if (add) {
      appendPlanOps(plan, compute);
      return;
    }
    for (PlanOp op : llvm::reverse(compute.ops)) {
      // All non-P operations in this incrementer are Hermitian.
      if (op.kind == PlanOpKind::P) {
        op.angle = -std::get<double>(op.angle);
      }
      plan.append(std::move(op));
    }
  };

  if (dirtyMode == Hp24DirtyMode::OneDirty) {
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

static SmallVector<Value> synthesizeRCCX(OpBuilder& builder, Location loc,
                                         Value control0, Value control1,
                                         Value target) {
  SmallVector<Value> wires = {control0, control1, target};
  GateEmitter(builder, loc, wires).emitRCCXSequence(0, 1, 2);
  return wires;
}

static CircuitPlan mczCoreForWidth(OpBuilder& builder, Location loc,
                                   size_t numControls);
static size_t entanglingCost(const CircuitPlan& plan);

static SmallVector<Value>
synthesizeControlledPauli(OpBuilder& builder, Location loc, ValueRange controls,
                          Value target, ControlledTarget gate) {
  SmallVector<Value> wires(controls);
  wires.push_back(target);
  const size_t k = controls.size();
  GateEmitter emitter(builder, loc, wires);
  // Small cores implement X; larger cores implement Z.
  const bool conjugate =
      k < 4 ? gate == ControlledTarget::Z : gate == ControlledTarget::X;
  if (conjugate) {
    emitter.h(k);
  }
  if (k == 2) {
    emitter.emitTwoControlledXSequence(0, 1, 2);
  } else if (k == 3) {
    emitter.emitThreeControlledXSequence();
  } else {
    lowerPlan(emitter, mczCoreForWidth(builder, loc, k));
  }
  if (conjugate) {
    emitter.h(k);
  }
  return wires;
}

//===----------------------------------------------------------------------===//
// Multi-controlled phase synthesis
//===----------------------------------------------------------------------===//

// Compare MCP(π) with HP24 through this width; linear HP24 is cheaper above it.
static constexpr size_t K_MCX_SP22_MAX_CONTROLS = 32;

//===----------------------------------------------------------------------===//
// Balanced-control rotations and phase correction
//===----------------------------------------------------------------------===//

static CircuitPlan planMultiControlledRz(OpBuilder& builder, Location loc,
                                         const mqt::FloatParameter& theta,
                                         size_t k);

/// Conjugate the entire RZ plan, including its phase-cancelling helpers.
static CircuitPlan changeRotationBasis(CircuitPlan rzPlan, size_t target,
                                       UnitaryOpInterface rotation) {
  const bool isY = isa<RYOp>(rotation.getOperation());
  const bool isX = isa<RXOp>(rotation.getOperation());
  if (!isX && !isY) {
    return rzPlan;
  }
  CircuitPlan plan;
  if (isY) {
    plan.append({.kind = PlanOpKind::P, .wires = {target}, .angle = -K_PI / 2});
  }
  plan.append({.kind = PlanOpKind::H, .wires = {target}});
  appendPlanOps(plan, std::move(rzPlan));
  plan.append({.kind = PlanOpKind::H, .wires = {target}});
  if (isY) {
    plan.append({.kind = PlanOpKind::P, .wires = {target}, .angle = K_PI / 2});
  }
  return plan;
}

static CircuitPlan planMultiControlledRotation(OpBuilder& builder, Location loc,
                                               const mqt::FloatParameter& theta,
                                               size_t numControls,
                                               UnitaryOpInterface rotation) {
  return changeRotationBasis(
      planMultiControlledRz(builder, loc, theta, numControls), numControls,
      rotation);
}

/// Optimized ancilla-free `C^2(P(θ))`.
///
/// Wires: `c0`, `c1`, target.
static CircuitPlan planMcpTwoControlled(OpBuilder& builder, Location loc,
                                        const mqt::FloatParameter& theta) {
  CircuitPlan plan;
  plan.ops.reserve(10);
  const auto quarter = mqt::scaleParameter(builder, loc, theta, 0.25);
  const auto negativeQuarter = mqt::scaleParameter(builder, loc, quarter, -1.0);
  const auto half = mqt::scaleParameter(builder, loc, theta, 0.5);
  constexpr size_t c0 = 0;
  constexpr size_t c1 = 1;
  constexpr size_t target = 2;
  plan.append({.kind = PlanOpKind::CX, .wires = {c0, target}});
  plan.append(
      {.kind = PlanOpKind::P, .wires = {target}, .angle = negativeQuarter});
  plan.append({.kind = PlanOpKind::CX, .wires = {c1, target}});
  plan.append({.kind = PlanOpKind::P, .wires = {target}, .angle = quarter});
  plan.append({.kind = PlanOpKind::CX, .wires = {c0, target}});
  plan.append(
      {.kind = PlanOpKind::P, .wires = {target}, .angle = negativeQuarter});
  plan.append({.kind = PlanOpKind::CX, .wires = {c1, target}});
  plan.append({.kind = PlanOpKind::P, .wires = {target}, .angle = quarter});
  plan.append({.kind = PlanOpKind::CRZ, .wires = {c0, c1}, .angle = half});
  plan.append({.kind = PlanOpKind::P, .wires = {c0}, .angle = quarter});
  return plan;
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
///
/// Controls `0..n-1`, target `n`; requires `n >= 5`.
static CircuitPlan planMcpSp22(OpBuilder& builder, Location loc,
                               const mqt::FloatParameter& theta,
                               size_t numControls) {
  CircuitPlan plan;
  const size_t n = numControls;
  const size_t target = n;
  plan.ops.reserve((2 * n * n) - (2 * n) + 1);

  const auto rootAngle = [&](double sign, size_t exponent) {
    if (const auto* constant = std::get_if<double>(&theta)) {
      return mqt::FloatParameter{
          std::ldexp(sign * *constant, -static_cast<int>(exponent))};
    }
    return mqt::scaleParameter(builder, loc, theta,
                               std::ldexp(sign, -static_cast<int>(exponent)));
  };

  // P_n(U)
  for (size_t c = 1; c < n; ++c) {
    plan.append({
        .kind = PlanOpKind::CP,
        .wires = {c, target},
        .angle = rootAngle(1.0, n - c),
    });
  }

  // Mid-root
  plan.append({
      .kind = PlanOpKind::CP,
      .wires = {0, target},
      .angle = rootAngle(1.0, n - 1),
  });

  // Q_n
  const CircuitPlan qn = buildSp22Q(n);
  plan.ops.append(qn.ops.begin(), qn.ops.end());

  // P_n(U)^dagger
  for (size_t c = 1; c < n; ++c) {
    plan.append({
        .kind = PlanOpKind::CP,
        .wires = {c, target},
        .angle = rootAngle(-1.0, n - c),
    });
  }

  // Q_n^dagger
  for (size_t i = qn.ops.size(); i-- > 0;) {
    const PlanOp& op = qn.ops[i];
    plan.append({
        .kind = op.kind,
        .wires = op.wires,
        .angle = -std::get<double>(op.angle),
    });
  }

  return plan;
}

static CircuitPlan planMcpForWidth(OpBuilder& builder, Location loc,
                                   const mqt::FloatParameter& theta, size_t k) {
  // The recursive RZ construction costs 6k²−26k+40 for k>=3; SP22 costs
  // 4k²−4k+2. Choose before emitting any symbolic parameter arithmetic.
  if (k >= 9) {
    return planMcpSp22(builder, loc, theta, k);
  }
  const auto scaledAngle = [&](size_t width) {
    return mqt::scaleParameter(builder, loc, theta,
                               std::ldexp(1.0, -static_cast<int>(k - width)));
  };
  CircuitPlan plan = planMcpTwoControlled(builder, loc, scaledAngle(2));
  // MCP(k, θ) = MCRZ(k, θ) MCP(k−1, θ/2) on the controls. Omitting the
  // second factor would lose a conditional scalar phase, not a global phase.
  for (size_t width = 3; width <= k; ++width) {
    appendPlanOps(
        plan, planMultiControlledRz(builder, loc, scaledAngle(width), width));
  }
  return plan;
}

/// Exact MCZ core, shared by Pauli/SWAP lowering and helper cost comparison.
static CircuitPlan mczCoreForWidth(OpBuilder& builder, Location loc,
                                   size_t numControls) {
  if (numControls < 9) {
    // Z = P(π), not RZ(π): the MCP planner retains the conditional scalar
    // phase. Its 32-CX k=4 plan also beats 20 CX plus seven CP (34
    // equivalents).
    return planMcpForWidth(builder, loc, K_PI, numControls);
  }
  auto incrementer = planHp24Core(numControls);
  if (numControls > K_MCX_SP22_MAX_CONTROLS) {
    return incrementer;
  }
  auto phase = planMcpForWidth(builder, loc, K_PI, numControls);
  return entanglingCost(incrementer) < entanglingCost(phase)
             ? std::move(incrementer)
             : std::move(phase);
}

/// Compress controls into zero helpers, apply the residual gate, then restore.
///
/// The residual gate preserves the controls and helper wires, so the relative
/// phases of the RCCX compute ladder cancel with its inverse. Leave at least
/// two controls: another compute/uncompute pair costs more than it saves.
static CircuitPlan planWithCleanHelpers(CircuitPlan residual, size_t k,
                                        size_t helpers) {
  CircuitPlan compute;
  compute.append({.kind = PlanOpKind::RCCX, .wires = {0, 1, k + 1}});
  for (size_t i = 1; i < helpers; ++i) {
    compute.append(
        {.kind = PlanOpKind::RCCX, .wires = {i + 1, k + i, k + i + 1}});
  }
  CircuitPlan plan = compute;
  SmallVector<size_t> map{k + helpers};
  for (size_t i = helpers + 1; i < k; ++i) {
    map.push_back(i);
  }
  map.push_back(k);
  appendRemapped(plan, std::move(residual), map);
  for (const auto& operation : llvm::reverse(compute.ops)) {
    plan.append(operation);
  }
  return plan;
}

/// Rotate using only the original controls as temporary dirty workspace.
///
/// Zindorf–Bose, arXiv:2404.02279, Lemmas 5–8: a conjugated C²(iZ)
/// implements each half-MCZ times a diagonal phase on non-target wires only.
/// Those phases commute with target rotations and cancel with the inverse
/// half-gates. Balanced halves supply at least count−2 opposite controls;
/// every borrowed control retains its full joint state, with no external
/// helper.
static CircuitPlan planMultiControlledRz(OpBuilder& builder, Location loc,
                                         const mqt::FloatParameter& theta,
                                         size_t k) {
  const auto halfToggle = [&](size_t begin, size_t count, bool inverse) {
    CircuitPlan plan;
    if (count == 1) {
      plan.append({.kind = PlanOpKind::CX, .wires = {begin, k}});
      return plan;
    }
    plan.append({.kind = PlanOpKind::H, .wires = {k}});
    SmallVector<size_t> borrowed;
    if (count > 2) {
      for (size_t c = 0; borrowed.size() < count - 2; ++c) {
        if (c < begin || c >= begin + count) {
          borrowed.push_back(c);
        }
      }
    }
    borrowed.push_back(k);
    CircuitPlan compute;
    for (size_t i = count - 2; i-- > 0;) {
      compute.append({
          .kind = PlanOpKind::RCCX,
          .wires = {begin + i + 2, borrowed[i + 1], borrowed[i]},
      });
    }
    appendPlanOps(plan, compute);
    // C²(iZ), or its inverse, costs four CX. Its scalar i phase is
    // control-dependent and is cancelled by the inverse half-gate.
    const double quarter = (inverse ? -1.0 : 1.0) * K_PI / 4;
    for (size_t repeat = 0; repeat < 2; ++repeat) {
      plan.append({.kind = PlanOpKind::CX, .wires = {begin, borrowed[0]}});
      plan.append(
          {.kind = PlanOpKind::P, .wires = {borrowed[0]}, .angle = quarter});
      plan.append({.kind = PlanOpKind::CX, .wires = {begin + 1, borrowed[0]}});
      plan.append(
          {.kind = PlanOpKind::P, .wires = {borrowed[0]}, .angle = -quarter});
    }
    for (const auto& operation : llvm::reverse(compute.ops)) {
      plan.append(operation);
    }
    plan.append({.kind = PlanOpKind::H, .wires = {k}});
    return plan;
  };
  const auto quarter = mqt::scaleParameter(builder, loc, theta, 0.25);
  const auto negativeQuarter = mqt::scaleParameter(builder, loc, quarter, -1.0);
  const size_t first = (k + 1) / 2;
  const size_t second = k / 2;
  CircuitPlan plan;
  for (size_t repeat = 0; repeat < 2; ++repeat) {
    appendPlanOps(plan, halfToggle(0, first, repeat != 0));
    plan.append(
        {.kind = PlanOpKind::RZ, .wires = {k}, .angle = negativeQuarter});
    appendPlanOps(plan, halfToggle(first, second, repeat != 0));
    plan.append({.kind = PlanOpKind::RZ, .wires = {k}, .angle = quarter});
  }
  return plan;
}

/// Rotate with k−2 arbitrary helpers, restoring their full joint state.
///
/// In the HP24 borrowed-MCX construction, U = T C T C toggles the target;
/// C is a basis permutation with phases, acting only on controls/helpers.
/// Replacing T by RCCX changes only diagonal phases. Thus
/// U† RZ(-θ/2) U RZ(θ/2) is controlled RZ(θ), and the two C
/// factors next to RZ cancel. Basis changes give RX and RY without assuming
/// that the target-dependent relative phases commute with those rotations.
static CircuitPlan planWithDirtyRzHelpers(OpBuilder& builder, Location loc,
                                          const mqt::FloatParameter& theta,
                                          size_t k) {
  const CircuitPlan carry = planBorrowedMcxCarry(k);
  const PlanOp toggle{.kind = PlanOpKind::RCCX, .wires = {k - 1, 2 * k - 2, k}};
  const auto half = mqt::scaleParameter(builder, loc, theta, 0.5);
  CircuitPlan plan;
  plan.append({.kind = PlanOpKind::RZ, .wires = {k}, .angle = half});
  plan.append(toggle);
  appendPlanOps(plan, carry);
  plan.append(toggle);
  plan.append({
      .kind = PlanOpKind::RZ,
      .wires = {k},
      .angle = mqt::scaleParameter(builder, loc, half, -1.0),
  });
  plan.append(toggle);
  appendPlanOps(plan, carry);
  plan.append(toggle);
  return plan;
}

/// Preserve the conditional scalar phase in MCP = MCRZ · MCP(θ/2).
///
/// Recurse to the four-control phase core. Every
/// layer reuses the same external helpers; the caller compares the entire
/// candidate with the fallback before emitting any parameter arithmetic.
static CircuitPlan planWithDirtyPhaseHelpers(OpBuilder& builder, Location loc,
                                             const mqt::FloatParameter& theta,
                                             size_t k) {
  const auto scaledAngle = [&](size_t width) {
    return mqt::scaleParameter(builder, loc, theta,
                               std::ldexp(1.0, -static_cast<int>(k - width)));
  };
  constexpr size_t base = 4;
  CircuitPlan plan = planMcpForWidth(builder, loc, scaledAngle(base), base);
  for (size_t width = base + 1; width <= k; ++width) {
    SmallVector<size_t> map;
    for (size_t i = 0; i <= width; ++i) {
      map.push_back(i);
    }
    for (size_t i = 0; i < width - 2; ++i) {
      map.push_back(k + 1 + i);
    }
    appendRemapped(
        plan, planWithDirtyRzHelpers(builder, loc, scaledAngle(width), width),
        map);
  }
  return plan;
}

/// Keep unused slots removable when their declared capacity exceeds the target.
static bool allocationsFitTarget(Operation* root, size_t capacity) {
  size_t allocated = 0;
  return !root->walk([&](Operation* op) {
                uint64_t size = 0;
                if (isa<AllocOp>(op)) {
                  size = 1;
                } else if (auto tensor = dyn_cast<qtensor::AllocOp>(op)) {
                  auto extent = getConstantIntValue(tensor.getSize());
                  if (!extent || *extent < 0) {
                    return WalkResult::interrupt();
                  }
                  size = static_cast<uint64_t>(*extent);
                }
                if (size > capacity - allocated) {
                  return WalkResult::interrupt();
                }
                allocated += size;
                return WalkResult::advance();
              })
              .wasInterrupted();
}

/// Follow earlier SSA definitions without recursing through tensor bookkeeping.
/// nullopt means unknown occupancy; false means occupied but not known zero.
static std::optional<bool> helperState(Value value, std::optional<int64_t> slot,
                                       const DenseSet<Value>& restoredZeros) {
  while (true) {
    auto* definition = value.getDefiningOp();
    if (!slot) {
      if (value.getDefiningOp<AllocOp>() || value.getDefiningOp<ResetOp>() ||
          restoredZeros.contains(value)) {
        return true;
      }
      if (auto extract = dyn_cast_or_null<qtensor::ExtractOp>(definition)) {
        slot = getConstantIntValue(extract.getIndex());
        if (slot) {
          value = extract.getTensor();
          continue;
        }
      }
      return false;
    }
    if (definition == nullptr) {
      return std::nullopt;
    }
    if (auto extract = dyn_cast<qtensor::ExtractOp>(definition)) {
      auto index = getConstantIntValue(extract.getIndex());
      if (!index || *index == *slot) {
        return std::nullopt;
      }
      value = extract.getTensor();
    } else if (auto insert = dyn_cast<qtensor::InsertOp>(definition)) {
      auto index = getConstantIntValue(insert.getIndex());
      if (!index) {
        return std::nullopt;
      }
      if (*index == *slot) {
        value = insert.getScalar();
        slot.reset();
      } else {
        value = insert.getDest();
      }
    } else if (isa<qtensor::AllocOp>(definition)) {
      return true;
    } else if (auto elements = dyn_cast<qtensor::FromElementsOp>(definition)) {
      value = elements.getElements()[*slot];
      slot.reset();
    } else {
      return std::nullopt;
    }
  }
}

static SmallVector<BorrowedHelper>
findHelpers(CtrlOp op, size_t limit, const DenseSet<Value>& restoredZeros) {
  SmallVector<BorrowedHelper> helpers, dirty;
  const auto remember = [&](BorrowedHelper helper) {
    auto& candidates = helper.clean ? helpers : dirty;
    if (candidates.size() < limit) {
      candidates.push_back(helper);
    }
  };
  auto* block = op->getBlock();
  const auto inspect = [&](Value value) {
    if (helpers.size() == limit) {
      return;
    }
    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!isa<QubitType>(value.getType()) &&
        !(tensorType && tensorType.getRank() == 1 &&
          tensorType.hasStaticShape() &&
          isa<QubitType>(tensorType.getElementType()))) {
      return;
    }
    auto* user = *value.getUsers().begin();
    // No captures, future definitions, consumed wires, or gate operands.
    if (user->getBlock() != block || !op->isBeforeInBlock(user)) {
      return;
    }
    if (!tensorType) {
      remember({
          .value = value,
          .slot = std::nullopt,
          .clean =
              helperState(value, std::nullopt, restoredZeros).value_or(false),
      });
      return;
    }
    for (int64_t slot = 0;
         slot < tensorType.getDimSize(0) && helpers.size() < limit; ++slot) {
      if (auto clean = helperState(value, slot, restoredZeros)) {
        remember({.value = value, .slot = slot, .clean = *clean});
      }
    }
  };
  for (auto argument : block->getArguments()) {
    inspect(argument);
  }
  for (auto& previous : *block) {
    if (&previous == op.getOperation() || helpers.size() == limit) {
      break;
    }
    for (auto result : previous.getResults()) {
      inspect(result);
    }
  }
  // Keep clean-first, source-order selection without storing unused slots.
  helpers.append(dirty.begin(), dirty.begin() + std::min(limit - helpers.size(),
                                                         dirty.size()));
  return helpers;
}

/// Exact balanced MCX, with one external helper. Each half restores its
/// internally borrowed opposite controls. A B A suffices for a zero helper;
/// A B A B cancels its unknown initial value for an arbitrary dirty helper.
static CircuitPlan planOneHelperMcx(size_t k, bool clean) {
  const size_t split = (k + 1) / 2;
  SmallVector<size_t> first, second;
  for (size_t i = 0; i < split; ++i) {
    first.push_back(i);
  }
  first.push_back(k + 1);
  for (size_t i = split; i < k; ++i) {
    first.push_back(i);
    second.push_back(i);
  }
  first.push_back(k);
  second.push_back(k + 1);
  second.push_back(k);
  for (size_t i = 0; i < split; ++i) {
    second.push_back(i);
  }
  CircuitPlan plan;
  appendRemapped(plan, planBorrowedHelperMcx(split), first);
  appendRemapped(plan, planBorrowedHelperMcx(k - split + 1), second);
  appendRemapped(plan, planBorrowedHelperMcx(split), first);
  if (!clean) {
    appendRemapped(plan, planBorrowedHelperMcx(k - split + 1), second);
  }
  return plan;
}

/// Pre-routing CX-equivalent estimate, not a promise about final mapped cost.
static size_t entanglingCost(const CircuitPlan& plan) {
  size_t count = 0;
  for (const auto& operation : plan.ops) {
    switch (operation.kind) {
    case PlanOpKind::CX:
      count += 1;
      break;
    case PlanOpKind::CP:
    case PlanOpKind::CRX:
    case PlanOpKind::CRZ:
      count += 2;
      break;
    case PlanOpKind::CCX:
      count += 6;
      break;
    case PlanOpKind::CCCX:
      count += 14;
      break;
    case PlanOpKind::RCCX:
      count += 3;
      break;
    case PlanOpKind::CCP:
      count += 6;
      break;
    case PlanOpKind::H:
    case PlanOpKind::X:
    case PlanOpKind::P:
    case PlanOpKind::RZ:
      break;
    }
  }
  return count;
}

static LogicalResult synthesizeWithHelpers(CtrlOp op, UnitaryOpInterface inner,
                                           PatternRewriter& rewriter,
                                           uint64_t budget,
                                           DenseSet<Value>& restoredZeros) {
  const bool swap = isa<SWAPOp>(inner.getOperation());
  const bool y = isa<YOp>(inner.getOperation());
  auto phase = dyn_cast<POp>(inner.getOperation());
  const bool rotation = isa<RXOp, RYOp, RZOp>(inner.getOperation());
  const auto theta = phase || rotation
                         ? mqt::foldParameter(inner.getParameters()[0])
                         : mqt::FloatParameter{0.0};
  const bool z = isa<ZOp>(inner.getOperation()) || (phase && isPiPhase(theta));
  const bool generalPhase = phase && !z;
  const bool parameterized = generalPhase || rotation;
  // SWAP's second target becomes one more control of the central MCX.
  const size_t k = op.getNumControls() + (swap ? 1 : 0);
  if (budget == 0 || k < 3) {
    return failure();
  }
  auto helpers =
      findHelpers(op, std::min<uint64_t>(budget, k - 2), restoredZeros);
  if (helpers.empty()) {
    return failure();
  }
  const size_t cleanCount = llvm::count_if(
      helpers, [](const BorrowedHelper& helper) { return helper.clean; });
  if (parameterized && cleanCount == 0 && k < 5) {
    return failure();
  }
  // Plan with a constant placeholder until acceptance: symbolic angle planning
  // emits arithmetic and a failed match must not mutate the IR.
  const auto parameterizedPlan = [&](const mqt::FloatParameter& angle,
                                     size_t controls) {
    return rotation ? planMultiControlledRotation(rewriter, op.getLoc(), angle,
                                                  controls, inner)
                    : planMcpForWidth(rewriter, op.getLoc(), angle, controls);
  };
  const auto dirtyPlan = [&](const mqt::FloatParameter& angle) {
    if (generalPhase) {
      return planWithDirtyPhaseHelpers(rewriter, op.getLoc(), angle, k);
    }
    return changeRotationBasis(
        planWithDirtyRzHelpers(rewriter, op.getLoc(), angle, k), k, inner);
  };
  auto best = parameterized ? parameterizedPlan(0.0, k)
                            : planOneHelperMcx(k, helpers.front().clean);
  const auto fallbackCost =
      parameterized
          ? entanglingCost(best)
          : (k == 3
                 ? 14
                 : entanglingCost(mczCoreForWidth(rewriter, op.getLoc(), k)));
  size_t used = parameterized ? 0 : 1;
  bool dirtyParameterized = false;
  if (parameterized && cleanCount != 0) {
    best = planWithCleanHelpers(parameterizedPlan(0.0, k - cleanCount), k,
                                cleanCount);
    used = cleanCount;
  }
  if (parameterized && k >= 5 && helpers.size() == k - 2) {
    auto dirty = dirtyPlan(0.0);
    if (entanglingCost(dirty) < entanglingCost(best)) {
      best = std::move(dirty);
      used = k - 2;
      dirtyParameterized = true;
    }
  }
  if (!parameterized && helpers.size() == k - 2) {
    auto many = helpers.back().clean
                    ? planWithCleanHelpers(planBorrowedHelperMcx(2), k, k - 2)
                    : planBorrowedHelperMcx(k);
    if (entanglingCost(many) < entanglingCost(best)) {
      best = std::move(many);
      used = helpers.size();
    }
  }
  if (entanglingCost(best) >= fallbackCost) {
    return failure();
  }
  if (parameterized) {
    mqt::hoistSupportingOpsBefore(*op.getBody(), inner.getOperation(), op,
                                  rewriter);
    rewriter.setInsertionPoint(op);
    best =
        dirtyParameterized
            ? dirtyPlan(theta)
            : planWithCleanHelpers(parameterizedPlan(theta, k - used), k, used);
  }
  helpers.resize(used);
  SmallVector<Value> wires(op.getControlsIn());
  if (swap) {
    wires.push_back(op.getInputTarget(1));
  }
  wires.push_back(op.getInputTarget(0));
  struct BorrowedValue {
    Value original;
    Value current;
    OpOperand* continuation;
  };
  SmallVector<BorrowedValue> borrowed;
  SmallVector<size_t> owners;
  for (const auto& helper : helpers) {
    auto* owner = llvm::find_if(borrowed, [&](const BorrowedValue& item) {
      return item.original == helper.value;
    });
    if (owner == borrowed.end()) {
      borrowed.push_back({
          .original = helper.value,
          .current = helper.value,
          .continuation = &*helper.value.getUses().begin(),
      });
      owner = std::prev(borrowed.end());
    }
    owners.push_back(std::distance(borrowed.begin(), owner));
    if (helper.slot) {
      auto index =
          arith::ConstantIndexOp::create(rewriter, op.getLoc(), *helper.slot);
      auto extract = qtensor::ExtractOp::create(rewriter, op.getLoc(),
                                                owner->current, index);
      owner->current = extract.getOutTensor();
      wires.push_back(extract.getResult());
    } else {
      wires.push_back(helper.value);
    }
  }
  GateEmitter emitter(rewriter, op.getLoc(), wires);
  if (swap) {
    emitter.cx(k, k - 1);
  } else if (y) {
    wires[k] = SdgOp::create(rewriter, op.getLoc(), wires[k]).getOutputQubit(0);
  } else if (z) {
    emitter.h(k);
  }
  lowerPlan(emitter, best);
  if (swap) {
    emitter.cx(k, k - 1);
  } else if (y) {
    wires[k] = SOp::create(rewriter, op.getLoc(), wires[k]).getOutputQubit(0);
  } else if (z) {
    emitter.h(k);
  }
  for (size_t i = 0; i < helpers.size(); ++i) {
    auto& owner = borrowed[owners[i]];
    if (helpers[i].clean) {
      restoredZeros.insert(wires[k + 1 + i]);
    }
    if (helpers[i].slot) {
      auto index = arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                  *helpers[i].slot);
      owner.current =
          qtensor::InsertOp::create(rewriter, op.getLoc(), wires[k + 1 + i],
                                    owner.current, index)
              .getResult();
    } else {
      owner.current = wires[k + 1 + i];
    }
  }
  for (auto& owner : borrowed) {
    rewriter.modifyOpInPlace(owner.continuation->getOwner(),
                             [&] { owner.continuation->set(owner.current); });
  }
  if (swap) {
    // CtrlOp results retain the original target order, not the MCX wire order.
    std::swap(wires[k - 1], wires[k]);
  }
  rewriter.replaceOp(op, ArrayRef<Value>(wires).take_front(k + 1));
  return success();
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

  SmallVector<Value> mcxControls(controls);
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

template <typename ModifierOp>
struct UnrollControlledModifier final : OpRewritePattern<ModifierOp> {
  UnrollControlledModifier(MLIRContext* context, uint64_t minQubits,
                           const CompilerTarget* target)
      : OpRewritePattern<ModifierOp>(context), minQubits_(minQubits),
        target_(target) {}

  LogicalResult matchAndRewrite(ModifierOp op,
                                PatternRewriter& rewriter) const override {
    auto control = op->template getParentOfType<CtrlOp>();
    if (!control || control.getNumQubits() < minQubits_ ||
        isWithinTargetNativeUnitary(op, target_)) {
      return failure();
    }
    return mqt::unrollModifier(op, rewriter);
  }

private:
  uint64_t minQubits_;
  const CompilerTarget* target_;
};

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
      return mqt::unrollModifier(op, rewriter);
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
    const bool rotation = isa<RXOp, RYOp, RZOp>(inner.getOperation());
    const bool phase = isa<POp>(inner.getOperation());
    const auto theta = rotation || phase
                           ? mqt::foldParameter(inner.getParameters()[0])
                           : mqt::FloatParameter{0.0};
    if (rotation || (phase && !isPiPhase(theta))) {
      // Preserve region-local symbolic angles before replacing their owner.
      mqt::hoistSupportingOpsBefore(*op.getBody(), inner.getOperation(), op,
                                    rewriter);
      rewriter.setInsertionPoint(op);
      SmallVector<Value> wires(op.getControlsIn());
      wires.push_back(op.getInputTarget(0));
      GateEmitter emitter(rewriter, op.getLoc(), wires);
      lowerPlan(emitter,
                rotation ? planMultiControlledRotation(
                               rewriter, op.getLoc(), theta, numControls, inner)
                         : planMcpForWidth(rewriter, op.getLoc(), theta,
                                           numControls));
      rewriter.replaceOp(op, wires);
      return success();
    }
    if (!phase && !isa<XOp, ZOp>(inner.getOperation())) {
      return failure();
    }
    // P(±π) is Z, including the conditional scalar phase.
    const auto gate = isa<XOp>(inner.getOperation()) ? ControlledTarget::X
                                                     : ControlledTarget::Z;
    rewriter.setInsertionPoint(op);
    rewriter.replaceOp(
        op, synthesizeControlledPauli(rewriter, op.getLoc(), op.getControlsIn(),
                                      op.getInputTarget(0), gate));
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

  DecomposeMultiControlled(const CompilerTarget& target, uint64_t minQubitsIn,
                           std::optional<uint64_t> ancillaBudgetIn)
      : target_(target) {
    minQubits = minQubitsIn;
    if (ancillaBudgetIn) {
      ancillaBudget = *ancillaBudgetIn;
    }
  }

protected:
  void runOnOperation() override {
    if (minQubits < 3) {
      getOperation().emitError()
          << "decompose-multi-controlled requires min-qubits >= 3";
      signalPassFailure();
      return;
    }

    bool hasStatic = false;
    getOperation().walk([&](StaticOp) { hasStatic = true; });
    const CompilerTarget* nativeTarget =
        target_ && (target_->connectivityKind() ==
                        CompilerTarget::Connectivity::Kind::AllToAll ||
                    hasStatic)
            ? &*target_
            : nullptr;

    // Consider only input gates. Recursively borrowing for generated half-MCXs
    // could exceed the original gate's external-helper budget.
    if (ancillaBudget != 0 &&
        (!target_ ||
         allocationsFitTarget(getOperation(), target_->numSites()))) {
      SmallVector<CtrlOp> candidates;
      getOperation().walk([&](CtrlOp op) {
        auto inner = mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
        if (op.getNumQubits() >= minQubits && inner &&
            ((op.getNumTargets() == 1 &&
              isa<XOp, YOp, ZOp, POp, RXOp, RYOp, RZOp>(
                  inner.getOperation())) ||
             (op.getNumTargets() == 2 && isa<SWAPOp>(inner.getOperation()))) &&
            !isWithinTargetNativeUnitary(op, nativeTarget)) {
          candidates.push_back(op);
        }
      });
      PatternRewriter rewriter(&getContext());
      // These proofs remain valid only during this input-gate sweep. Discard
      // them before greedy rewriting can erase or recycle the SSA values.
      DenseSet<Value> restoredZeros;
      for (auto op : candidates) {
        rewriter.setInsertionPoint(op);
        auto inner = mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
        std::ignore = synthesizeWithHelpers(op, inner, rewriter, ancillaBudget,
                                            restoredZeros);
      }
    }
    RewritePatternSet patterns(&getContext());
    patterns
        .add<DecomposeControlledGatePattern, DecomposeRCCXPattern,
             UnrollControlledModifier<InvOp>, UnrollControlledModifier<PowOp>>(
            &getContext(), minQubits, nativeTarget);
    CtrlOp::getOptimizationPatterns(patterns, &getContext());
    InvOp::getLoweringPatterns(patterns, &getContext());
    PowOp::getLoweringPatterns(patterns, &getContext());
    if (ancillaBudget != 0) {
      // Borrowing can introduce inserts before later extracts. Restore the
      // flat tensor access form required by placement and routing.
      qtensor::ExtractOp::getCanonicalizationPatterns(patterns, &getContext());
      qtensor::InsertOp::getCanonicalizationPatterns(patterns, &getContext());
    }

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

private:
  std::optional<CompilerTarget> target_;
};

} // namespace

std::unique_ptr<Pass>
createDecomposeMultiControlled(const CompilerTarget& target, uint64_t minQubits,
                               std::optional<uint64_t> ancillaBudget) {
  return std::make_unique<DecomposeMultiControlled>(target, minQubits,
                                                    ancillaBudget);
}

} // namespace mlir::qco
