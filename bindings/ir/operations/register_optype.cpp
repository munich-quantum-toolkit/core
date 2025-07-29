/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/OpType.hpp"

// These includes must be the first includes for any bindings code
// clang-format off
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)

#include <pybind11/native_enum.h>
#include <pybind11/pytypes.h>
// clang-format on

#include <string>

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerOptype(const py::module& m) {
  py::native_enum<qc::OpType>(m, "OpType", "enum.Enum", "Test string.")
      .value("none", qc::OpType::None)
      .value("gphase", qc::OpType::GPhase)
      .value("i", qc::OpType::I)
      .value("h", qc::OpType::H)
      .value("x", qc::OpType::X)
      .value("y", qc::OpType::Y)
      .value("z", qc::OpType::Z)
      .value("s", qc::OpType::S)
      .value("sdg", qc::OpType::Sdg)
      .value("t", qc::OpType::T)
      .value("tdg", qc::OpType::Tdg)
      .value("v", qc::OpType::V)
      .value("vdg", qc::OpType::Vdg)
      .value("u", qc::OpType::U)
      .value("u2", qc::OpType::U2)
      .value("p", qc::OpType::P)
      .value("sx", qc::OpType::SX)
      .value("sxdg", qc::OpType::SXdg)
      .value("rx", qc::OpType::RX)
      .value("ry", qc::OpType::RY)
      .value("rz", qc::OpType::RZ)
      .value("swap", qc::OpType::SWAP)
      .value("iswap", qc::OpType::iSWAP)
      .value("iswapdg", qc::OpType::iSWAPdg)
      .value("peres", qc::OpType::Peres)
      .value("peresdg", qc::OpType::Peresdg)
      .value("dcx", qc::OpType::DCX)
      .value("ecr", qc::OpType::ECR)
      .value("rxx", qc::OpType::RXX)
      .value("ryy", qc::OpType::RYY)
      .value("rzz", qc::OpType::RZZ)
      .value("rzx", qc::OpType::RZX)
      .value("xx_minus_yy", qc::OpType::XXminusYY)
      .value("xx_plus_yy", qc::OpType::XXplusYY)
      .value("compound", qc::OpType::Compound)
      .value("measure", qc::OpType::Measure)
      .value("reset", qc::OpType::Reset)
      .value("barrier", qc::OpType::Barrier)
      .value("classic_controlled", qc::OpType::ClassicControlled)
      .export_values()
      .finalize();
}

} // namespace mqt
