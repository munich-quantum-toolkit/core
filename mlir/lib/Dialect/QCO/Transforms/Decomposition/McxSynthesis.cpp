/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/MultiControlled.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <numbers>
#include <numeric>
#include <utility>

namespace mlir::qco::decomposition {

namespace {

constexpr double K_PI = std::numbers::pi;
constexpr double K_PI8 = K_PI / 8.0;

/// Emits QCO gates for multi-controlled X decomposition.
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

  void h(std::size_t q) {
    setWire(q, HOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void x(std::size_t q) {
    setWire(q, XOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void p(std::size_t q, double theta) {
    setWire(q, POp::create(*builder_, loc_, wire(q), theta).getOutputQubit(0));
  }

  void cx(std::size_t control, std::size_t target) {
    auto ctrlOp = CtrlOp::create(
        *builder_, loc_, ValueRange{wire(control)}, ValueRange{wire(target)},
        [&](ValueRange args) -> SmallVector<Value> {
          return {XOp::create(*builder_, loc_, args[0]).getOutputQubit(0)};
        });
    setWire(control, ctrlOp.getControlsOut()[0]);
    setWire(target, ctrlOp.getTargetsOut()[0]);
  }

  void cp(std::size_t control, std::size_t target, double theta) {
    p(control, theta / 2.0);
    cx(control, target);
    p(target, -theta / 2.0);
    cx(control, target);
    p(target, theta / 2.0);
  }

  void t(std::size_t q) {
    setWire(q, TOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void tdg(std::size_t q) {
    setWire(q, TdgOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void ccp(double theta, std::size_t c0, std::size_t c1, std::size_t target) {
    cx(c0, target);
    p(target, -theta / 4.0);
    cx(c1, target);
    p(target, theta / 4.0);
    cx(c0, target);
    p(target, -theta / 4.0);
    cx(c1, target);
    p(target, theta / 4.0);
    p(c0, theta / 4.0);
    p(c1, theta / 4.0);
    cx(c0, c1);
    p(c1, -theta / 4.0);
    cx(c0, c1);
  }

  /// Standard CCX decomposition.
  void emitCcx(std::size_t c0, std::size_t c1, std::size_t target) {
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

  /// Relative-phase CCX.
  void emitRelativeCcx(std::size_t c0, std::size_t c1, std::size_t target) {
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

  void c3x() {
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

} // namespace

static void synthRelativeMcx(GateEmitter& builder, std::size_t numControls);

static void addActionGadget(GateEmitter& builder, std::size_t q0,
                            std::size_t q1, std::size_t q2) {
  builder.h(q2);
  builder.t(q2);
  builder.cx(q0, q2);
  builder.tdg(q2);
  builder.cx(q1, q2);
}

static void addResetGadget(GateEmitter& builder, std::size_t q0, std::size_t q1,
                           std::size_t q2) {
  builder.cx(q1, q2);
  builder.t(q2);
  builder.cx(q0, q2);
  builder.tdg(q2);
  builder.h(q2);
}

static void synthMcxNDirtyI15(GateEmitter& builder, std::size_t numControls) {
  if (numControls == 1) {
    builder.cx(0, 1);
  } else if (numControls == 2) {
    builder.emitCcx(0, 1, 2);
  } else if (numControls == 3) {
    builder.c3x();
  } else {
    const std::size_t controlsEnd = numControls;
    const std::size_t target = numControls;
    const std::size_t firstAncilla = numControls + 1;
    const std::size_t lastAncilla = firstAncilla + numControls - 3;
    for (std::size_t pass = 0; pass < 2; ++pass) {
      builder.emitCcx(controlsEnd - 1, lastAncilla, target);
      for (std::size_t i = numControls - 3; i-- > 0;) {
        addActionGadget(builder, i + 2, firstAncilla + i, firstAncilla + i + 1);
      }
      builder.emitRelativeCcx(0, 1, firstAncilla);
      for (std::size_t i = 0; i < numControls - 3; ++i) {
        addResetGadget(builder, i + 2, firstAncilla + i, firstAncilla + i + 1);
      }
    }
  }
}

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
    builder.emitRelativeCcx(0, 1, 2);
    return;
  }

  const std::size_t num3 = numControls / 3;
  const std::size_t num2 = (numControls - num3) / 2;
  const std::size_t num1 = numControls - num3 - num2;
  const std::size_t block2Begin = num1;
  const std::size_t block3Begin = num1 + num2;
  const std::size_t controlsEnd = numControls;

  SmallVector<std::size_t, 16> map;
  const auto relativeStep = [&](const double sign, const std::size_t begin,
                                const std::size_t end, const std::size_t k) {
    builder.p(target, sign * K_PI8);
    map.clear();
    for (std::size_t q = begin; q < end; ++q) {
      map.push_back(q);
    }
    map.push_back(target);
    builder.compose(map, [&](GateEmitter& sub) { synthRelativeMcx(sub, k); });
  };

  builder.h(target);
  relativeStep(+1, block3Begin, controlsEnd, num3);
  relativeStep(-1, block2Begin, block3Begin, num2);
  relativeStep(+1, block3Begin, controlsEnd, num3);
  relativeStep(-1, 0, block2Begin, num1);
  relativeStep(+1, block3Begin, controlsEnd, num3);
  relativeStep(-1, block2Begin, block3Begin, num2);
  relativeStep(+1, block3Begin, controlsEnd, num3);
  relativeStep(-1, 0, block2Begin, num1);
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

static void incrementDirty(GateEmitter& builder, std::size_t n,
                           std::size_t numDirtyAncillae, bool flagAdd) {
  if (numDirtyAncillae == 1 && n % 2 == 0) {
    return;
  }

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

/// HP24 no-auxiliary MCX core, without target Hadamard bookends.
/// @param n Total wire count (controls plus target).
static void emitMcxHp24Core(GateEmitter& emitter, std::size_t n) {
  const std::size_t lastControl = n - 1;
  const std::size_t c0 = n - 2;
  const std::size_t c1 = n - 1;

  SmallVector<std::size_t, 16> incrementQubits(n);
  std::iota(incrementQubits.begin(), incrementQubits.end(), 0U);

  // One dirty ancilla for very large even widths (22+ controls); two otherwise.
  if ((n % 2 == 0) && (n >= 23)) {
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

SmallVector<Value> synthesizeMcx(OpBuilder& builder, Location loc,
                                 ValueRange controls, Value target) {
  if (controls.size() < 2) {
    llvm::reportFatalUsageError(
        "synthesizeMcx requires at least 2 control qubits");
  }

  SmallVector<Value> wires(controls.begin(), controls.end());
  wires.push_back(target);

  const std::size_t n = controls.size() + 1;
  const std::size_t targetIdx = controls.size();

  GateEmitter emitter(builder, loc, wires);
  emitter.h(targetIdx);
  emitMcxHp24Core(emitter, n);
  emitter.h(targetIdx);
  return wires;
}

SmallVector<Value> synthesizeMcz(OpBuilder& builder, Location loc,
                                 ValueRange controls, Value target) {
  if (controls.size() < 2) {
    llvm::reportFatalUsageError(
        "synthesizeMcz requires at least 2 control qubits");
  }

  SmallVector<Value> wires(controls.begin(), controls.end());
  wires.push_back(target);

  const std::size_t n = controls.size() + 1;

  GateEmitter emitter(builder, loc, wires);
  // Algebraically MCZ = H·(H·CORE·H)·H = CORE for k >= 2.
  emitMcxHp24Core(emitter, n);
  return wires;
}

} // namespace mlir::qco::decomposition
