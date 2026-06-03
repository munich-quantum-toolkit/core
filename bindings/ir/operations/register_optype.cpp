/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/OpType.hpp"

#include <nanobind/nanobind.h>

namespace mqt {

namespace nb = nanobind;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerOptype(const nb::module_& m) {
  nb::enum_<qc::OpType>(m, "OpType", "Enumeration of operation types.")

      .value("none", qc::OpType::None, R"pb(A placeholder operation.

It is used to represent an operation that is not yet defined.)pb")

      .value("gphase", qc::OpType::GPhase, R"pb(A global phase operation.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.gphase`)pb")

      .value("i", qc::OpType::I, R"pb(An identity operation.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.i`)pb")

      .value("h", qc::OpType::H, R"pb(A Hadamard gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.h`)pb")

      .value("x", qc::OpType::X, R"pb(An X gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.x`)pb")

      .value("y", qc::OpType::Y, R"pb(A Y gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.y`)pb")

      .value("z", qc::OpType::Z, R"pb(A Z gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.z`)pb")

      .value("s", qc::OpType::S, R"pb(An S gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.s`)pb")

      .value("sdg", qc::OpType::Sdg, R"pb(An :math:`S^\dagger` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.sdg`)pb")

      .value("t", qc::OpType::T, R"pb(A T gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.t`)pb")

      .value("tdg", qc::OpType::Tdg, R"pb(A :math:`T^\dagger` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.tdg`)pb")

      .value("v", qc::OpType::V, R"pb(A V gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.v`)pb")

      .value("vdg", qc::OpType::Vdg, R"pb(A :math:`V^\dagger` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.vdg`)pb")

      .value("u", qc::OpType::U, R"pb(A U gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.u`)pb")

      .value("u2", qc::OpType::U2, R"pb(A U2 gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.u2`)pb")

      .value("p", qc::OpType::P, R"pb(A phase gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.p`)pb")

      .value("sx", qc::OpType::SX, R"pb(A :math:`\sqrt{X}` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.sx`)pb")

      .value("sxdg", qc::OpType::SXdg, R"pb(A :math:`\sqrt{X}^\dagger` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.sxdg`)pb")

      .value("rx", qc::OpType::RX, R"pb(A :math:`R_x` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.rx`)pb")

      .value("ry", qc::OpType::RY, R"pb(A :math:`R_y` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.ry`)pb")

      .value("rz", qc::OpType::RZ, R"pb(A :math:`R_z` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.rz`)pb")

      .value("r", qc::OpType::R, R"pb(An :math:`R` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.r`)pb")

      .value("swap", qc::OpType::SWAP, R"pb(A SWAP gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.swap`)pb")

      .value("iswap", qc::OpType::iSWAP, R"pb(A iSWAP gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.iswap`)pb")

      .value("iswapdg", qc::OpType::iSWAPdg,
             R"pb(A :math:`i\text{SWAP}^\dagger` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.iswapdg`)pb")

      .value("peres", qc::OpType::Peres, R"pb(A Peres gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.peres`)pb")

      .value("peresdg", qc::OpType::Peresdg,
             R"pb(A :math:`\text{Peres}^\dagger` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.peresdg`)pb")

      .value("dcx", qc::OpType::DCX, R"pb(A DCX gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.dcx`)pb")

      .value("ecr", qc::OpType::ECR, R"pb(An ECR gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.ecr`)pb")

      .value("rxx", qc::OpType::RXX, R"pb(A :math:`R_{xx}` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.rxx`)pb")

      .value("ryy", qc::OpType::RYY, R"pb(A :math:`R_{yy}` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.ryy`)pb")

      .value("rzz", qc::OpType::RZZ, R"pb(A :math:`R_{zz}` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.rzz`)pb")

      .value("rzx", qc::OpType::RZX, R"pb(A :math:`R_{zx}` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.rzx`)pb")

      .value("xx_minus_yy", qc::OpType::XXminusYY,
             R"pb(A :math:`R_{XX - YY}` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.xx_minus_yy`)pb")

      .value("xx_plus_yy", qc::OpType::XXplusYY,
             R"pb(A :math:`R_{XX + YY}` gate.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.xx_plus_yy`)pb")

      .value("compound", qc::OpType::Compound, R"pb(A compound operation.

It is used to group multiple operations into a single operation.

See also :class:`.CompoundOperation`)pb")

      .value("measure", qc::OpType::Measure, R"pb(A measurement operation.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.measure`)pb")

      .value("reset", qc::OpType::Reset, R"pb(A reset operation.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.reset`)pb")

      .value("barrier", qc::OpType::Barrier, R"pb(A barrier operation.

It is used to separate operations in the circuit.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.barrier`)pb")

      .value("if_else", qc::OpType::IfElse, R"pb(An if-else operation.

It is used to control the execution of an operation based on the value of a classical register.

See Also:
    :meth:`mqt.core.ir.QuantumComputation.if_else`)pb");
}

} // namespace mqt
