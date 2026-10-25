/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/Pauli.hpp"
#include "dd/StateGeneration.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dd {
namespace {

fp densePauliExpectation(const CVec& state, const std::string_view paulis) {
  std::complex<fp> result = 0.;
  for (size_t column = 0; column < state.size(); ++column) {
    auto row = column;
    std::complex<fp> phase = 1.;
    for (size_t qubit = 0; qubit < paulis.size(); ++qubit) {
      const auto bit = (column >> qubit) & 1U;
      switch (paulis[paulis.size() - qubit - 1]) {
      case 'X':
        row ^= size_t{1} << qubit;
        break;
      case 'Y':
        row ^= size_t{1} << qubit;
        phase *= std::complex<fp>{0., bit == 0 ? 1. : -1.};
        break;
      case 'Z':
        phase *= bit == 0 ? 1. : -1.;
        break;
      default:
        break;
      }
    }
    result += std::conj(state[row]) * phase * state[column];
  }
  EXPECT_NEAR(result.imag(), 0., 1e-12);
  return result.real();
}

TEST(PauliExpectations, AllThreeQubitStringsAgainstDenseOracle) {
  Package package(3);
  CVec amplitudes{
      {.1, .2},  {.3, -.4}, {-.2, .3}, {.4, .1},
      {.2, -.1}, {-.3, .2}, {.1, -.3}, {.4, -.2},
  };
  const auto state = makeStateFromVector(amplitudes, package);
  const auto roots = package.getRootSet<vNode>();
  std::vector<PauliObservable> observables;
  constexpr std::string_view alphabet = "IXYZ";
  for (size_t label = 0; label < 64; ++label) {
    std::string paulis(3, 'I');
    for (size_t index = 0; index < 3; ++index) {
      paulis[index] = alphabet[(label >> (2 * index)) & 3U];
    }
    observables.push_back({
        {.paulis = paulis, .coefficient = .75},
        {.paulis = "III", .coefficient = -.125},
        {.paulis = paulis, .coefficient = -.25},
    });
  }
  const auto values = package.expectationValues(state, 3, observables);
  ASSERT_EQ(values.size(), observables.size());
  for (size_t index = 0; index < values.size(); ++index) {
    EXPECT_NEAR(
        values[index],
        (.5 * densePauliExpectation(amplitudes, observables[index][0].paulis)) -
            (.125 * densePauliExpectation(amplitudes, "III")),
        1e-12);
  }
  EXPECT_EQ(package.getRootSet<vNode>(), roots);
  EXPECT_TRUE(package.getRootSet<mNode>().empty());
  const auto phase = std::polar(1., .37);
  for (auto& amplitude : amplitudes) {
    amplitude *= phase;
  }
  const auto phased = makeStateFromVector(amplitudes, package);
  const auto phasedValues = package.expectationValues(phased, 3, observables);
  for (size_t index = 0; index < values.size(); ++index) {
    EXPECT_NEAR(values[index], phasedValues[index], 1e-12);
  }
  package.decRef(phased);
  package.decRef(state);
  EXPECT_TRUE(package.getRootSet<vNode>().empty());
}

TEST(PauliExpectations, ImaginaryPhaseAndQubitOrder) {
  Package package(2);
  const auto state = makeStateFromVector(
      {0., 0., SQRT2_2, std::complex<fp>{0., SQRT2_2}}, package);
  const std::array<PauliObservable, 5> observables{
      PauliObservable{
          {.paulis = "IY", .coefficient = 1.}, // spellchecker:disable-line
      },
      PauliObservable{{.paulis = "YI", .coefficient = 1.}},
      PauliObservable{{.paulis = "ZI", .coefficient = 1.}},
      PauliObservable{
          {.paulis = "IZ", .coefficient = 1.}, // spellchecker:disable-line
      },
      PauliObservable{{.paulis = "ZY", .coefficient = 1.}},
  };
  const auto values = package.expectationValues(state, 2, observables);
  const std::array<fp, 5> expected{1., 0., -1., 0., -1.};
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(values[index], expected[index], 1e-12);
  }
  package.decRef(state);
}

TEST(PauliExpectations, EmptyZeroIdentityAndScalarStates) {
  Package package(4);
  const auto scalar = makeStateFromVector({std::complex<fp>{2., 1.}}, package);
  const std::array<PauliObservable, 3> scalars{
      PauliObservable{},
      PauliObservable{{.paulis = "", .coefficient = 3.}},
      PauliObservable{
          {.paulis = "", .coefficient = 2.},
          {.paulis = "", .coefficient = -2.},
      },
  };
  EXPECT_EQ(package.expectationValues(scalar, 0, scalars),
            (std::vector<fp>{0., 15., 0.}));
  EXPECT_TRUE(package.expectationValues(scalar, 0, {}).empty());
  const std::array<PauliObservable, 2> observables{
      PauliObservable{{.paulis = "XYZI", .coefficient = 2.}},
      PauliObservable{{.paulis = "IIII", .coefficient = 3.}},
  };
  EXPECT_EQ(package.expectationValues(vEdge::zero(), 4, observables),
            (std::vector<fp>{0., 0.}));
  package.decRef(scalar);
}

TEST(PauliExpectations, InvalidInputsLeaveBorrowedStateLive) {
  Package package(2);
  const auto state = makeZeroState(2, package);
  const auto roots = package.getRootSet<vNode>();
  const std::array<PauliObservable, 1> valid{
      PauliObservable{{.paulis = "II", .coefficient = 1.}},
  };
  EXPECT_THROW(package.expectationValues(state, 1, valid),
               std::invalid_argument);
  EXPECT_THROW(package.expectationValues(state, 3, valid),
               std::invalid_argument);
  for (const auto& term : std::vector<PauliTerm>{
           {.paulis = "X", .coefficient = 1.},
           {.paulis = "XXI", .coefficient = 1.},
           {.paulis = "IA", .coefficient = 1.},
           {.paulis = "ix", .coefficient = 1.},
           {.paulis = "II", .coefficient = std::numeric_limits<fp>::infinity()},
           {
               .paulis = "II",
               .coefficient = std::numeric_limits<fp>::quiet_NaN(),
           },
       }) {
    const std::array<PauliObservable, 1> invalid{PauliObservable{term}};
    EXPECT_THROW(package.expectationValues(state, 2, invalid),
                 std::invalid_argument);
  }
  Package other(2);
  EXPECT_THROW(other.expectationValues(state, 2, valid), std::invalid_argument);
  EXPECT_EQ(package.getRootSet<vNode>(), roots);
  EXPECT_TRUE(package.getRootSet<mNode>().empty());
  package.decRef(state);
  EXPECT_THROW(package.expectationValues(state, 2, valid),
               std::invalid_argument);
}

TEST(PauliExpectations, OverflowLeavesBorrowedStateLive) {
  Package package(1);
  const auto state = makeZeroState(1, package);
  const auto roots = package.getRootSet<vNode>();
  const auto maximum = std::numeric_limits<fp>::max();
  const std::array<PauliObservable, 2> observables{
      PauliObservable{{.paulis = "X", .coefficient = 1.}},
      PauliObservable{
          {.paulis = "I", .coefficient = maximum},
          {.paulis = "I", .coefficient = maximum},
      },
  };
  EXPECT_THROW(package.expectationValues(state, 1, observables),
               std::overflow_error);
  EXPECT_EQ(package.getRootSet<vNode>(), roots);
  EXPECT_TRUE(package.getRootSet<mNode>().empty());
  package.decRef(state);

  const auto scalar =
      makeStateFromVector({std::complex<fp>{1e200, 0.}}, package);
  const auto scalarRoots = package.getRootSet<vNode>();
  const std::array<PauliObservable, 1> scalarIdentity{
      PauliObservable{{.paulis = "", .coefficient = 1.}},
  };
  EXPECT_THROW(package.expectationValues(scalar, 0, scalarIdentity),
               std::overflow_error);
  EXPECT_EQ(package.getRootSet<vNode>(), scalarRoots);
  package.decRef(scalar);
}

TEST(PauliExpectations, CompactSixtyFourQubitGHZ) {
  constexpr size_t width = 64;
  Package package(width);
  const auto state = makeGHZState(width, package);
  const auto roots = package.getRootSet<vNode>();
  auto endpoints = std::string(width, 'I');
  endpoints.front() = 'Z';
  endpoints.back() = 'Z';
  auto single = std::string(width, 'I');
  single.back() = 'Z';
  const std::array<PauliObservable, 4> observables{
      PauliObservable{{.paulis = std::string(width, 'X'), .coefficient = 1.}},
      PauliObservable{{.paulis = endpoints, .coefficient = 1.}},
      PauliObservable{{.paulis = single, .coefficient = 1.}},
      PauliObservable{
          {.paulis = endpoints, .coefficient = 2.},
          {.paulis = endpoints, .coefficient = -1.},
      },
  };
  const auto values = package.expectationValues(state, width, observables);
  const std::array<fp, 4> expected{1., 1., 0., 1.};
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(values[index], expected[index], 1e-12);
  }
  EXPECT_EQ(package.getRootSet<vNode>(), roots);
  EXPECT_TRUE(package.getRootSet<mNode>().empty());
  package.decRef(state);
}

} // namespace
} // namespace dd
