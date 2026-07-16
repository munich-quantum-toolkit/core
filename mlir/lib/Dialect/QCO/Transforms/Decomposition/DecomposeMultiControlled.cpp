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

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h> // IWYU pragma: keep (Passes.h.inc)
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <numbers>
#include <numeric>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_DECOMPOSEMULTICONTROLLED
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

enum class ControlledTarget : std::uint8_t { X, Z, Phase };

constexpr double K_PI = std::numbers::pi;
constexpr double K_PI8 = K_PI / 8.0;

class GateEmitter {
public:
  GateEmitter(OpBuilder& builder, Location loc, SmallVector<Value>& wires,
              ArrayRef<std::size_t> remap = {})
      : builder_(&builder), loc_(loc), wires_(&wires), remap_(remap) {}

  template <typename Fn> void compose(ArrayRef<std::size_t> qubitMap, Fn&& fn) {
    SmallVector<std::size_t, 16> composeRemap;
    composeRemap.reserve(qubitMap.size());
    for (std::size_t local : qubitMap) {
      composeRemap.push_back(wireIndex(local));
    }
    GateEmitter nested(*builder_, loc_, *wires_, composeRemap);
    std::forward<Fn>(fn)(nested);
  }

  // Single- and two-qubit primitives
  void h(std::size_t q) {
    setWire(q, HOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void x(std::size_t q) {
    setWire(q, XOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void p(std::size_t q, double theta) {
    setWire(q, POp::create(*builder_, loc_, wire(q), theta).getOutputQubit(0));
  }

  void t(std::size_t q) {
    setWire(q, TOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void tdg(std::size_t q) {
    setWire(q, TdgOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void cx(std::size_t control, std::size_t target) {
    auto ctrlOp = CtrlOp::create(
        *builder_, loc_, wire(control), wire(target),
        [&](Value targetArg) -> Value {
          return XOp::create(*builder_, loc_, targetArg).getOutputQubit(0);
        });
    setWire(control, ctrlOp.getControlsOut()[0]);
    setWire(target, ctrlOp.getTargetsOut()[0]);
  }

  void cp(std::size_t control, std::size_t target, double theta) {
    auto ctrlOp =
        CtrlOp::create(*builder_, loc_, wire(control), wire(target),
                       [&](Value targetArg) -> Value {
                         return POp::create(*builder_, loc_, targetArg, theta)
                             .getOutputQubit(0);
                       });
    setWire(control, ctrlOp.getControlsOut()[0]);
    setWire(target, ctrlOp.getTargetsOut()[0]);
  }

  void crz(std::size_t control, std::size_t target, double theta) {
    const double half = theta / 2.0;
    p(target, half);
    cx(control, target);
    p(target, -half);
    cx(control, target);
  }

  // Controlled-RX via RX(theta) = H RZ(theta) H, reusing crz.
  void crx(std::size_t control, std::size_t target, double theta) {
    h(target);
    crz(control, target, theta);
    h(target);
  }

  // Building blocks left as QCO ops (further lowered by min-controls)
  void emitCcx(std::size_t c0, std::size_t c1, std::size_t target) {
    emitCtrl({c0, c1}, target, [](OpBuilder& builder, Location loc, Value arg) {
      return XOp::create(builder, loc, arg).getOutputQubit(0);
    });
  }

  // Multi-controlled X of arbitrary width, left as a single CtrlOp for the
  // greedy driver to decompose recursively (used by the v24 phase recursion).
  void mcx(ArrayRef<std::size_t> controls, std::size_t target) {
    if (controls.size() == 1) {
      cx(controls[0], target);
      return;
    }
    emitCtrl(controls, target, [](OpBuilder& builder, Location loc, Value arg) {
      return XOp::create(builder, loc, arg).getOutputQubit(0);
    });
  }

  void emitThreeControlledX(std::size_t c0, std::size_t c1, std::size_t c2,
                            std::size_t target) {
    emitCtrl({c0, c1, c2}, target,
             [](OpBuilder& builder, Location loc, Value arg) {
               return XOp::create(builder, loc, arg).getOutputQubit(0);
             });
  }

  void emitRCCX(std::size_t c0, std::size_t c1, std::size_t target) {
    auto rccxOp =
        RCCXOp::create(*builder_, loc_, wire(c0), wire(c1), wire(target));
    setWire(c0, rccxOp.getOutputQubit(0));
    setWire(c1, rccxOp.getOutputQubit(1));
    setWire(target, rccxOp.getOutputQubit(2));
  }

  void ccp(double theta, std::size_t c0, std::size_t c1, std::size_t target) {
    emitCtrl({c0, c1}, target,
             [theta](OpBuilder& builder, Location loc, Value arg) {
               return POp::create(builder, loc, arg, theta).getOutputQubit(0);
             });
  }

  // Fully expanded elementary sequences
  void emitRCCXSequence(std::size_t c0, std::size_t c1, std::size_t target) {
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

  void emitTwoControlledXSequence(std::size_t c0, std::size_t c1,
                                  std::size_t target) {
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

  void emitTwoControlledPhaseSequence(std::size_t c0, std::size_t c1,
                                      std::size_t target, double theta) {
    const double quarter = theta / 4.0;
    const double half = theta / 2.0;
    cx(c0, target);
    p(target, -quarter);
    cx(c1, target);
    p(target, quarter);
    cx(c0, target);
    p(target, -quarter);
    cx(c1, target);
    p(target, quarter);
    crz(c0, c1, half);
    p(c0, quarter);
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

  // Dirty-ancilla gadgets for HP24
  void addGadget(std::size_t q0, std::size_t q1, std::size_t q2, bool invert) {
    if (!invert) {
      emitGadgetBody(*this, q0, q1, q2);
      return;
    }
    SmallVector<Value> invWires = {wire(q0), wire(q1), wire(q2)};
    auto invOp = InvOp::create(
        *builder_, loc_, invWires, [&](ValueRange args) -> SmallVector<Value> {
          SmallVector<Value> local(args.begin(), args.end());
          GateEmitter inner(*builder_, loc_, local);
          emitGadgetBody(inner, 0, 1, 2);
          return local;
        });
    assignWires({invOp.getOutputQubit(0), invOp.getOutputQubit(1),
                 invOp.getOutputQubit(2)},
                {q0, q1, q2});
  }

private:
  static void emitGadgetBody(GateEmitter& builder, std::size_t q0,
                             std::size_t q1, std::size_t q2) {
    builder.h(q2);
    builder.t(q2);
    builder.cx(q0, q2);
    builder.tdg(q2);
    builder.cx(q1, q2);
  }

  void emitCtrl(ArrayRef<std::size_t> controls, std::size_t target,
                llvm::function_ref<Value(OpBuilder&, Location, Value)> body) {
    SmallVector<Value> controlValues;
    controlValues.reserve(controls.size());
    for (std::size_t control : controls) {
      controlValues.push_back(wire(control));
    }
    auto ctrlOp = CtrlOp::create(*builder_, loc_, controlValues, wire(target),
                                 [&](Value targetArg) -> Value {
                                   return body(*builder_, loc_, targetArg);
                                 });
    for (std::size_t i = 0; i < controls.size(); ++i) {
      setWire(controls[i], ctrlOp.getControlsOut()[i]);
    }
    setWire(target, ctrlOp.getTargetsOut()[0]);
  }

  void assignWires(ArrayRef<Value> values,
                   std::initializer_list<std::size_t> indices) {
    std::size_t i = 0;
    for (std::size_t index : indices) {
      setWire(index, values[i++]);
    }
  }

  [[nodiscard]] std::size_t wireIndex(std::size_t local) const {
    return remap_.empty() ? local : remap_[local];
  }

  [[nodiscard]] Value wire(std::size_t local) const {
    return (*wires_)[wireIndex(local)];
  }

  void setWire(std::size_t local, Value value) {
    (*wires_)[wireIndex(local)] = value;
  }

  OpBuilder* builder_;
  Location loc_;
  SmallVector<Value>* wires_;
  ArrayRef<std::size_t> remap_;
};

struct ControlledGateSpec {
  ControlledTarget gate;
  std::optional<double> theta;
};

} // namespace

//===----------------------------------------------------------------------===//
// HP24 low-level helpers
//===----------------------------------------------------------------------===//

static void ux(GateEmitter& builder, std::size_t q1, std::size_t q2,
               std::size_t q3) {
  builder.cx(q1, q3);
  builder.cx(q1, q2);
  builder.emitCcx(q2, q3, q1);
}

static void uz(GateEmitter& builder, std::size_t q1, std::size_t q2,
               std::size_t q3) {
  builder.emitCcx(q2, q3, q1);
  builder.cx(q1, q2);
  builder.cx(q2, q3);
}

//===----------------------------------------------------------------------===//
// n-dirty relative / absolute MCX
//===----------------------------------------------------------------------===//

static void synthMcxNDirtyI15(GateEmitter& builder, std::size_t numControls) {
  if (numControls == 1) {
    builder.cx(0, 1);
  } else if (numControls == 2) {
    builder.emitCcx(0, 1, 2);
  } else if (numControls == 3) {
    builder.emitThreeControlledX(0, 1, 2, 3);
  } else {
    const std::size_t controlsEnd = numControls;
    const std::size_t target = numControls;
    const std::size_t firstAncilla = numControls + 1;
    const std::size_t lastAncilla = firstAncilla + numControls - 3;
    for (std::size_t pass = 0; pass < 2; ++pass) {
      builder.emitCcx(controlsEnd - 1, lastAncilla, target);
      for (std::size_t i = numControls - 3; i-- > 0;) {
        builder.addGadget(i + 2, firstAncilla + i, firstAncilla + i + 1, false);
      }
      builder.emitRCCX(0, 1, firstAncilla);
      for (std::size_t i = 0; i < numControls - 3; ++i) {
        builder.addGadget(i + 2, firstAncilla + i, firstAncilla + i + 1, true);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Dirty increment
//===----------------------------------------------------------------------===//

static void incrementNDirtyLarge(GateEmitter& builder, std::size_t n) {
  const std::size_t lastQubit = n - 1;

  builder.x(n);
  for (std::size_t q = 0; q < n; ++q) {
    builder.cx(n, q);
  }
  builder.x(n);

  for (std::size_t i = 0; i < n - 1; ++i) {
    ux(builder, n, n + 1 + i, i);
  }

  builder.cx(n, lastQubit);
  for (std::size_t i = n - 1; i-- > 0;) {
    uz(builder, n, n + 1 + i, i);
  }

  for (std::size_t i = 0; i < n - 1; ++i) {
    builder.x(n + 1 + i);
  }

  for (std::size_t i = 0; i < n - 1; ++i) {
    ux(builder, n, n + 1 + i, i);
  }
  builder.cx(n, lastQubit);
  for (std::size_t i = n - 1; i-- > 0;) {
    uz(builder, n, n + 1 + i, i);
  }
  for (std::size_t i = 0; i < n - 1; ++i) {
    builder.x(n + 1 + i);
  }

  builder.x(lastQubit);
  builder.x(n);
  for (std::size_t q = 0; q < n; ++q) {
    builder.cx(n, q);
  }
  builder.x(n);
}

static void incrementNDirtySmall(GateEmitter& builder, std::size_t n) {
  SmallVector<std::size_t, 16> qubits;
  for (std::size_t k = n - 1; k >= 1; --k) {
    qubits.clear();
    for (std::size_t q = 0; q <= k; ++q) {
      qubits.push_back(q);
    }
    for (std::size_t q = n + 1; q < 2 * n; ++q) {
      qubits.push_back(q);
    }
    builder.compose(qubits,
                    [&](GateEmitter& sub) { synthMcxNDirtyI15(sub, k); });
  }
  builder.x(0);
}

static void incrementNDirty(GateEmitter& builder, std::size_t n) {
  if (n <= 10) {
    incrementNDirtySmall(builder, n);
  } else {
    incrementNDirtyLarge(builder, n);
  }
}

//===----------------------------------------------------------------------===//
// Relative-phase MCX
//===----------------------------------------------------------------------===//

static void synthRelativeMcx(GateEmitter& builder, std::size_t numControls) {
  const std::size_t target = numControls;

  if (numControls == 0) {
    return;
  }
  if (numControls == 1) {
    builder.cx(0, 1);
    return;
  }
  if (numControls == 2) {
    builder.emitRCCX(0, 1, 2);
    return;
  }

  const std::size_t num3 = numControls / 3;
  const std::size_t num2 = (numControls - num3) / 2;
  const std::size_t num1 = numControls - num3 - num2;
  const std::size_t block2Begin = num1;
  const std::size_t block3Begin = num1 + num2;
  const std::size_t controlsEnd = numControls;

  SmallVector<std::size_t, 16> map;
  const auto relativeStep = [&](const std::size_t begin, const std::size_t end,
                                const std::size_t k,
                                const bool positive = true) {
    builder.p(target, positive ? K_PI8 : -K_PI8);
    map.clear();
    for (std::size_t q = begin; q < end; ++q) {
      map.push_back(q);
    }
    map.push_back(target);
    builder.compose(map, [&](GateEmitter& sub) { synthRelativeMcx(sub, k); });
  };

  builder.h(target);
  relativeStep(block3Begin, controlsEnd, num3);
  relativeStep(block2Begin, block3Begin, num2, false);
  relativeStep(block3Begin, controlsEnd, num3);
  relativeStep(0, block2Begin, num1, false);
  relativeStep(block3Begin, controlsEnd, num3);
  relativeStep(block2Begin, block3Begin, num2, false);
  relativeStep(block3Begin, controlsEnd, num3);
  relativeStep(0, block2Begin, num1, false);
  builder.h(target);
}

static void synthRelativeMcxNDirty(GateEmitter& builder,
                                   std::size_t numControls) {
  if (numControls < 11) {
    synthRelativeMcx(builder, numControls);
  } else {
    synthMcxNDirtyI15(builder, numControls);
  }
}

//===----------------------------------------------------------------------===//
// Dirty-ancilla increment (HP24)
//===----------------------------------------------------------------------===//

static void incrementDirty(GateEmitter& builder, std::size_t n,
                           std::size_t numDirtyAncillae, bool flagAdd) {
  const std::size_t k = numDirtyAncillae == 1 ? (n + 1) / 2 : (n + 2) / 2;
  const std::size_t ancilla1 = n;
  const std::size_t ancilla2 = n + 1;
  const std::size_t incrementWidth = numDirtyAncillae == 1 ? k : (1 + n - k);

  if (!flagAdd) {
    for (std::size_t i = 0; i < n; ++i) {
      builder.x(i);
    }
  }

  SmallVector<std::size_t, 16> k12Qubits;
  k12Qubits.push_back(ancilla1);
  for (std::size_t q = k; q < n; ++q) {
    k12Qubits.push_back(q);
  }
  for (std::size_t q = 0; q < k; ++q) {
    k12Qubits.push_back(q);
  }
  if (numDirtyAncillae == 2) {
    k12Qubits.push_back(ancilla2);
  }

  builder.compose(k12Qubits, [&](GateEmitter& sub) {
    incrementNDirty(sub, incrementWidth);
  });
  builder.x(ancilla1);
  for (std::size_t q = k; q < n; ++q) {
    builder.cx(ancilla1, q);
  }

  SmallVector<std::size_t, 16> kMcxQubits;
  for (std::size_t q = 0; q < k; ++q) {
    kMcxQubits.push_back(q);
  }
  kMcxQubits.push_back(ancilla1);
  for (std::size_t q = k; q < n; ++q) {
    kMcxQubits.push_back(q);
  }
  if (numDirtyAncillae == 2) {
    kMcxQubits.push_back(ancilla2);
  }

  builder.compose(kMcxQubits,
                  [&](GateEmitter& sub) { synthRelativeMcxNDirty(sub, k); });
  builder.compose(k12Qubits, [&](GateEmitter& sub) {
    incrementNDirty(sub, incrementWidth);
  });
  builder.x(ancilla1);
  builder.compose(kMcxQubits,
                  [&](GateEmitter& sub) { synthRelativeMcxNDirty(sub, k); });
  for (std::size_t q = k; q < n; ++q) {
    builder.cx(ancilla1, q);
  }

  SmallVector<std::size_t, 16> k3Qubits;
  for (std::size_t q = 0; q < k; ++q) {
    k3Qubits.push_back(q);
  }
  for (std::size_t q = k; q < n; ++q) {
    k3Qubits.push_back(q);
  }
  k3Qubits.push_back(ancilla1);
  if (numDirtyAncillae == 2) {
    k3Qubits.push_back(ancilla2);
  }
  builder.compose(k3Qubits, [&](GateEmitter& sub) { incrementNDirty(sub, k); });

  if (!flagAdd) {
    for (std::size_t i = 0; i < n; ++i) {
      builder.x(i);
    }
  }
}

//===----------------------------------------------------------------------===//
// HP24 multi-controlled core
//===----------------------------------------------------------------------===//

static void emitMcxHp24Core(GateEmitter& emitter, std::size_t n) {
  const std::size_t lastControl = n - 1;
  const std::size_t c0 = n - 2;
  const std::size_t c1 = n - 1;

  SmallVector<std::size_t, 16> incrementQubits(n);
  std::iota(incrementQubits.begin(), incrementQubits.end(), 0U);

  const std::size_t numControls = n - 1;
  // One dirty ancilla for odd control counts >= 23 (even total wire count n);
  // two dirty ancillae otherwise.
  constexpr std::size_t kOneDirtyAncillaMinControls = 23;
  const bool useOneDirtyAncilla =
      numControls >= kOneDirtyAncillaMinControls && (numControls % 2 == 1);
  if (useOneDirtyAncilla) {
    emitter.compose(incrementQubits, [&](GateEmitter& sub) {
      incrementDirty(sub, n - 1, 1, true);
    });
    double phi = -K_PI;
    for (std::size_t q = n - 2; q > 0; --q) {
      phi /= 2.0;
      emitter.cp(q, lastControl, phi);
    }
    emitter.compose(incrementQubits, [&](GateEmitter& sub) {
      incrementDirty(sub, n - 1, 1, false);
    });
    phi = K_PI;
    for (std::size_t q = n - 2; q > 0; --q) {
      phi /= 2.0;
      emitter.cp(q, lastControl, phi);
    }
    emitter.cp(0, lastControl, phi);
  } else {
    emitter.compose(incrementQubits, [&](GateEmitter& sub) {
      incrementDirty(sub, n - 2, 2, true);
    });
    double phi = -K_PI;
    for (std::size_t q = n - 3; q > 0; --q) {
      phi /= 2.0;
      emitter.ccp(phi, q, c0, c1);
    }
    emitter.compose(incrementQubits, [&](GateEmitter& sub) {
      incrementDirty(sub, n - 2, 2, false);
    });
    phi = K_PI;
    for (std::size_t q = n - 3; q > 0; --q) {
      phi /= 2.0;
      emitter.ccp(phi, q, c0, c1);
    }
    emitter.ccp(phi, 0, c0, c1);
  }
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
                        std::optional<double> theta = std::nullopt) {
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
    emitter.emitTwoControlledPhaseSequence(0, 1, 2, theta.value());
    break;
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

static SmallVector<Value>
synthesizeMultiControlled(OpBuilder& builder, Location loc, ValueRange controls,
                          Value target, ControlledTarget gate) {
  SmallVector<Value> wires(controls.begin(), controls.end());
  wires.push_back(target);

  const std::size_t targetIdx = controls.size();
  GateEmitter emitter(builder, loc, wires);
  if (gate == ControlledTarget::X) {
    emitter.h(targetIdx);
    emitMcxHp24Core(emitter, wires.size());
    emitter.h(targetIdx);
  } else {
    emitMcxHp24Core(emitter, wires.size());
  }
  return wires;
}

//===----------------------------------------------------------------------===//
// Multi-controlled phase synthesis
//===----------------------------------------------------------------------===//

// Number of controls at which the linear-depth SP22 synthesis becomes more
// CX-efficient than the recursive v24 synthesis: v24 uses fewer CX gates for
// two to four controls, SP22 for five or more.
constexpr std::size_t K_MCP_SP22_MIN_CONTROLS = 5;

// Recursive multi-controlled phase synthesis (referred to as "v24"), using the
// textbook cp + mcx recursion (see Barenco et al., "Elementary gates for
// quantum computation", Phys. Rev. A 52, 3457 (1995)).
// Wires: controls 0..numControls-1, target numControls.
static void emitMcpV24(GateEmitter& emitter, double phi,
                       std::size_t numControls) {
  if (numControls == 1) {
    emitter.cp(0, 1, phi);
    return;
  }

  const std::size_t lastControl = numControls - 1;
  const std::size_t target = numControls;

  SmallVector<std::size_t, 16> innerControls;
  innerControls.reserve(lastControl);
  for (std::size_t c = 0; c < lastControl; ++c) {
    innerControls.push_back(c);
  }

  emitter.cp(lastControl, target, phi / 2.0);
  emitter.mcx(innerControls, lastControl);
  emitter.cp(lastControl, target, -phi / 2.0);
  emitter.mcx(innerControls, lastControl);

  // Recurse on the reduced control set with the same target. Remap so the
  // sub-emitter's target index (numControls - 1) refers to the real target.
  SmallVector<std::size_t, 16> recurseMap(innerControls.begin(),
                                          innerControls.end());
  recurseMap.push_back(target);
  emitter.compose(recurseMap, [&](GateEmitter& sub) {
    emitMcpV24(sub, phi / 2.0, numControls - 1);
  });
}

/// One of the four ordered passes of the SP22 linear-depth multi-controlled
/// phase synthesis.
///
/// Implements the multi-controlled phase construction of A. J. da Silva and
/// D. K. Park, "Linear-depth quantum circuits for multiqubit controlled gates",
/// Phys. Rev. A 106, 042602 (2022).
///
/// @note Adapted from the `MCPhaseGate` synthesis in the IBM Qiskit framework.
///       (C) Copyright IBM 2026
///
///       This code is licensed under the Apache License, Version 2.0. You may
///       obtain a copy of this license in the LICENSE.txt file in the root
///       directory of this source tree or at
///       https://www.apache.org/licenses/LICENSE-2.0.
///
///       Any modifications or derivative works of this code must retain this
///       copyright notice, and modified files need to carry a notice
///       indicating that they have been altered from the originals.
static void emitMcpSp22Step(GateEmitter& emitter, double phi,
                            std::size_t numQubits, int step) {
  const bool reverse = step == 1 || step == 3;
  const std::size_t start = reverse ? 0 : 1;

  SmallVector<std::pair<std::size_t, std::size_t>, 32> pairs;
  for (std::size_t target = 0; target < numQubits; ++target) {
    for (std::size_t control = start; control < target; ++control) {
      pairs.emplace_back(control, target);
    }
  }
  std::ranges::stable_sort(pairs, [reverse](const auto& a, const auto& b) {
    const std::size_t sumA = a.first + a.second;
    const std::size_t sumB = b.first + b.second;
    return reverse ? sumA > sumB : sumA < sumB;
  });

  for (const auto& [control, target] : pairs) {
    std::int64_t exponent =
        static_cast<std::int64_t>(target) - static_cast<std::int64_t>(control);
    if (control == 0) {
      exponent -= 1;
    }
    const double param = std::pow(2.0, static_cast<double>(exponent));

    if (target == numQubits - 1 && (step == 1 || step == 2)) {
      const double sign = step == 1 ? 1.0 : -1.0;
      emitter.cp(control, target, sign * phi / param);
      continue;
    }

    double sign = 1.0;
    if (step == 1) {
      sign = 1.0;
    } else if (step == 2) {
      sign = -1.0;
    } else if (step == 3) {
      sign = control == 0 ? -1.0 : 1.0;
    } else {
      sign = control == 0 ? 1.0 : -1.0;
    }
    emitter.crx(control, target, sign * K_PI / param);
  }
}

// SP22 linear-depth multi-controlled phase. Wires: controls 0..numControls-1,
// target numControls.
static void emitMcpSp22(GateEmitter& emitter, double phi,
                        std::size_t numControls) {
  const std::size_t numQubits = numControls + 1;
  emitMcpSp22Step(emitter, phi, numQubits, 1);
  emitMcpSp22Step(emitter, phi, numQubits, 2);
  emitMcpSp22Step(emitter, phi, numQubits - 1, 3);
  emitMcpSp22Step(emitter, phi, numQubits - 1, 4);
}

// Choose the CX-optimal synthesis for the given control count.
static void emitMcpDefault(GateEmitter& emitter, double phi,
                           std::size_t numControls) {
  if (numControls < K_MCP_SP22_MIN_CONTROLS) {
    emitMcpV24(emitter, phi, numControls);
  } else {
    emitMcpSp22(emitter, phi, numControls);
  }
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

namespace {

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
      rewriter.replaceOp(op, synthesizeTwoControlled(
                                 rewriter, op.getLoc(), op.getControlsIn()[0],
                                 op.getControlsIn()[1], op.getInputTarget(0),
                                 spec->gate, spec->theta));
      return success();
    }

    ControlledTarget gate = spec->gate;
    // A compile-time phase of +/- pi is exactly Z; route it through the more
    // efficient multi-controlled-Z (HP24 core) path.
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

} // namespace

} // namespace mlir::qco
