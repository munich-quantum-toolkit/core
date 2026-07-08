/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/MultiControlled.h"

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <numbers>
#include <numeric>
#include <optional>
#include <utility>

namespace mlir::qco::decomposition {

namespace {

constexpr double K_PI = std::numbers::pi;
constexpr double K_PI8 = K_PI / 8.0;

using Complex = std::complex<double>;

class GateEmitter {
public:
  GateEmitter(OpBuilder& builder, Location loc, SmallVector<Value>& wires,
              std::uint64_t minControls = UINT64_MAX,
              ArrayRef<std::size_t> remap = {})
      : builder_(&builder), loc_(loc), wires_(&wires), remap_(remap),
        minControls_(minControls) {}

  template <typename Fn> void compose(ArrayRef<std::size_t> qubitMap, Fn&& fn) {
    SmallVector<std::size_t, 16> composeRemap;
    composeRemap.reserve(qubitMap.size());
    for (std::size_t local : qubitMap) {
      composeRemap.push_back(wireIndex(local));
    }
    GateEmitter nested(*builder_, loc_, *wires_, minControls_, composeRemap);
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

  void rz(std::size_t q, double theta) {
    setWire(q, RZOp::create(*builder_, loc_, wire(q), theta).getOutputQubit(0));
  }

  void t(std::size_t q) {
    setWire(q, TOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
  }

  void tdg(std::size_t q) {
    setWire(q, TdgOp::create(*builder_, loc_, wire(q)).getOutputQubit(0));
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
    auto ctrlOp = CtrlOp::create(
        *builder_, loc_, ValueRange{wire(control)}, ValueRange{wire(target)},
        [&](ValueRange args) -> SmallVector<Value> {
          return {
              POp::create(*builder_, loc_, args[0], theta).getOutputQubit(0)};
        });
    setWire(control, ctrlOp.getControlsOut()[0]);
    setWire(target, ctrlOp.getTargetsOut()[0]);
  }

  void crz(std::size_t control, std::size_t target, double theta) {
    const double half = theta / 2.0;
    rz(target, half);
    cx(control, target);
    rz(target, -half);
    cx(control, target);
  }

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
    rz(target, -quarter);
    cx(c1, target);
    rz(target, quarter);
    cx(c0, target);
    rz(target, -quarter);
    cx(c1, target);
    rz(target, quarter);
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

  void emitCcx(std::size_t c0, std::size_t c1, std::size_t target) {
    if (minControls_ <= 2) {
      SmallVector<Value> local = {wire(c0), wire(c1), wire(target)};
      GateEmitter inner(*builder_, loc_, local);
      inner.emitTwoControlledXSequence(0, 1, 2);
      assignWires(local, {c0, c1, target});
      return;
    }
    emitCtrl({c0, c1}, target, [](OpBuilder& builder, Location loc, Value arg) {
      return XOp::create(builder, loc, arg).getOutputQubit(0);
    });
  }

  void emitThreeControlledX(std::size_t c0, std::size_t c1, std::size_t c2,
                            std::size_t target) {
    if (minControls_ <= 3) {
      SmallVector<Value> local = {wire(c0), wire(c1), wire(c2), wire(target)};
      GateEmitter inner(*builder_, loc_, local);
      inner.emitThreeControlledXSequence();
      assignWires(local, {c0, c1, c2, target});
      return;
    }
    emitCtrl({c0, c1, c2}, target,
             [](OpBuilder& builder, Location loc, Value arg) {
               return XOp::create(builder, loc, arg).getOutputQubit(0);
             });
  }

  void emitRCCX(std::size_t c0, std::size_t c1, std::size_t target) {
    if (minControls_ <= 2) {
      SmallVector<Value> local = {wire(c0), wire(c1), wire(target)};
      GateEmitter inner(*builder_, loc_, local);
      inner.emitRCCXSequence(0, 1, 2);
      assignWires(local, {c0, c1, target});
      return;
    }
    auto rccxOp =
        RCCXOp::create(*builder_, loc_, wire(c0), wire(c1), wire(target));
    setWire(c0, rccxOp.getOutputQubit(0));
    setWire(c1, rccxOp.getOutputQubit(1));
    setWire(target, rccxOp.getOutputQubit(2));
  }

  void ccp(double theta, std::size_t c0, std::size_t c1, std::size_t target) {
    if (minControls_ <= 2) {
      SmallVector<Value> local = {wire(c0), wire(c1), wire(target)};
      GateEmitter inner(*builder_, loc_, local);
      inner.emitTwoControlledPhaseSequence(0, 1, 2, theta);
      assignWires(local, {c0, c1, target});
      return;
    }
    emitCtrl({c0, c1}, target,
             [theta](OpBuilder& builder, Location loc, Value arg) {
               return POp::create(builder, loc, arg, theta).getOutputQubit(0);
             });
  }

private:
  void emitCtrl(ArrayRef<std::size_t> controls, std::size_t target,
                llvm::function_ref<Value(OpBuilder&, Location, Value)> body) {
    SmallVector<Value> controlValues;
    controlValues.reserve(controls.size());
    for (std::size_t control : controls) {
      controlValues.push_back(wire(control));
    }
    auto ctrlOp =
        CtrlOp::create(*builder_, loc_, controlValues, ValueRange{wire(target)},
                       [&](ValueRange args) -> SmallVector<Value> {
                         return {body(*builder_, loc_, args[0])};
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
  std::uint64_t minControls_;
};

[[nodiscard]] DynamicMatrix twoControlledXMatrix() {
  DynamicMatrix matrix = DynamicMatrix::identity(8);
  matrix.setBottomRightCorner(XOp::getUnitaryMatrix());
  return matrix;
}

[[nodiscard]] std::optional<double>
extractTwoControlledPhaseAngle(const DynamicMatrix& unitary) {
  if (unitary.rows() != 8 || unitary.cols() != 8) {
    return std::nullopt;
  }
  for (std::int64_t row = 0; row < 8; ++row) {
    for (std::int64_t col = 0; col < 8; ++col) {
      if (row == col) {
        if (row == 7) {
          continue;
        }
        if (std::abs(unitary(row, col) - Complex{1.0, 0.0}) >
            10.0 * MATRIX_TOLERANCE) {
          return std::nullopt;
        }
      } else if (std::abs(unitary(row, col)) > 10.0 * MATRIX_TOLERANCE) {
        return std::nullopt;
      }
    }
  }
  return std::arg(unitary(7, 7));
}

[[nodiscard]] SmallVector<Value> threeControlledWires(ValueRange controls,
                                                      Value target) {
  if (controls.size() != 3) {
    llvm::reportFatalUsageError(
        "three-controlled synthesis requires exactly 3 control qubits");
  }
  SmallVector<Value> wires(controls.begin(), controls.end());
  wires.push_back(target);
  return wires;
}

void addActionGadget(GateEmitter& builder, std::size_t q0, std::size_t q1,
                     std::size_t q2) {
  builder.h(q2);
  builder.t(q2);
  builder.cx(q0, q2);
  builder.tdg(q2);
  builder.cx(q1, q2);
}

void addResetGadget(GateEmitter& builder, std::size_t q0, std::size_t q1,
                    std::size_t q2) {
  builder.cx(q1, q2);
  builder.t(q2);
  builder.cx(q0, q2);
  builder.tdg(q2);
  builder.h(q2);
}

void synthMcxNDirtyI15(GateEmitter& builder, std::size_t numControls) {
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
        addActionGadget(builder, i + 2, firstAncilla + i, firstAncilla + i + 1);
      }
      builder.emitRCCX(0, 1, firstAncilla);
      for (std::size_t i = 0; i < numControls - 3; ++i) {
        addResetGadget(builder, i + 2, firstAncilla + i, firstAncilla + i + 1);
      }
    }
  }
}

void ux(GateEmitter& builder, std::size_t q1, std::size_t q2, std::size_t q3) {
  builder.cx(q1, q3);
  builder.cx(q1, q2);
  builder.emitCcx(q2, q3, q1);
}

void uz(GateEmitter& builder, std::size_t q1, std::size_t q2, std::size_t q3) {
  builder.emitCcx(q2, q3, q1);
  builder.cx(q1, q2);
  builder.cx(q2, q3);
}

void incrementNDirtyLarge(GateEmitter& builder, std::size_t n) {
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

void incrementNDirtySmall(GateEmitter& builder, std::size_t n) {
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

void incrementNDirty(GateEmitter& builder, std::size_t n) {
  if (n <= 10) {
    incrementNDirtySmall(builder, n);
  } else {
    incrementNDirtyLarge(builder, n);
  }
}

void synthRelativeMcx(GateEmitter& builder, std::size_t numControls) {
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

void synthRelativeMcxNDirty(GateEmitter& builder, std::size_t numControls) {
  if (numControls < 11) {
    synthRelativeMcx(builder, numControls);
  } else {
    synthMcxNDirtyI15(builder, numControls);
  }
}

void incrementDirty(GateEmitter& builder, std::size_t n,
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

void emitMcxHp24Core(GateEmitter& emitter, std::size_t n) {
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

} // namespace

SmallVector<Value> synthesizeRCCX(OpBuilder& builder, Location loc,
                                  Value control0, Value control1,
                                  Value target) {
  SmallVector<Value> wires = {control0, control1, target};
  GateEmitter(builder, loc, wires).emitRCCXSequence(0, 1, 2);
  return wires;
}

SmallVector<Value> synthesizeTwoControlled(OpBuilder& builder, Location loc,
                                           Value control0, Value control1,
                                           Value target, ControlledTarget gate,
                                           std::optional<double> theta) {
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
    if (!theta) {
      llvm::reportFatalUsageError(
          "synthesizeTwoControlled: phase gate requires theta");
    }
    emitter.emitTwoControlledPhaseSequence(0, 1, 2, *theta);
    break;
  }
  return wires;
}

SmallVector<Value> synthesizeTwoControlled(OpBuilder& builder, Location loc,
                                           Value control0, Value control1,
                                           Value target,
                                           const DynamicMatrix& unitary) {
  if (unitary.rows() != 8 || unitary.cols() != 8) {
    llvm::reportFatalUsageError(
        "synthesizeTwoControlled requires an 8x8 unitary matrix");
  }

  if (unitary.isApprox(twoControlledXMatrix())) {
    return synthesizeTwoControlled(builder, loc, control0, control1, target,
                                   ControlledTarget::X);
  }
  if (const auto phaseTheta = extractTwoControlledPhaseAngle(unitary)) {
    return synthesizeTwoControlled(builder, loc, control0, control1, target,
                                   ControlledTarget::Phase, *phaseTheta);
  }

  llvm::reportFatalUsageError(
      "synthesizeTwoControlled: unsupported two-controlled unitary");
}

SmallVector<Value> synthesizeThreeControlled(OpBuilder& builder, Location loc,
                                             ValueRange controls, Value target,
                                             ControlledTarget gate,
                                             std::optional<double> theta) {
  SmallVector<Value> wires = threeControlledWires(controls, target);
  GateEmitter emitter(builder, loc, wires);
  switch (gate) {
  case ControlledTarget::X:
    emitter.emitThreeControlledXSequence();
    break;
  case ControlledTarget::Z:
    emitter.h(3);
    emitter.emitThreeControlledXSequence();
    emitter.h(3);
    break;
  case ControlledTarget::Phase: {
    if (!theta) {
      llvm::reportFatalUsageError(
          "synthesizeThreeControlled: phase gate requires theta");
    }
    auto ctrlOp = CtrlOp::create(
        builder, loc, ValueRange{wires[0], wires[1], wires[2]},
        ValueRange{wires[3]}, [&](ValueRange args) -> SmallVector<Value> {
          return {POp::create(builder, loc, args[0], *theta).getOutputQubit(0)};
        });
    wires[0] = ctrlOp.getControlsOut()[0];
    wires[1] = ctrlOp.getControlsOut()[1];
    wires[2] = ctrlOp.getControlsOut()[2];
    wires[3] = ctrlOp.getTargetsOut()[0];
    break;
  }
  }
  return wires;
}

SmallVector<Value> synthesizeMultiControlled(OpBuilder& builder, Location loc,
                                             ValueRange controls, Value target,
                                             const std::uint64_t minControls,
                                             ControlledTarget gate,
                                             std::optional<double> theta) {
  if (controls.size() < 3) {
    llvm::reportFatalUsageError(
        "synthesizeMultiControlled requires at least 3 control qubits");
  }
  if (controls.size() == 3) {
    return synthesizeThreeControlled(builder, loc, controls, target, gate,
                                     theta);
  }
  if (gate == ControlledTarget::Phase) {
    llvm::reportFatalUsageError(
        "synthesizeMultiControlled: phase gates with four or more controls "
        "are not supported");
  }

  SmallVector<Value> wires(controls.begin(), controls.end());
  wires.push_back(target);

  const std::size_t targetIdx = controls.size();
  GateEmitter emitter(builder, loc, wires, minControls);
  if (gate == ControlledTarget::X) {
    emitter.h(targetIdx);
    emitMcxHp24Core(emitter, wires.size());
    emitter.h(targetIdx);
  } else {
    emitMcxHp24Core(emitter, wires.size());
  }
  return wires;
}

} // namespace mlir::qco::decomposition
