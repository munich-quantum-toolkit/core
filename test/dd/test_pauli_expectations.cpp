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

#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include "llvm/ADT/StringRef.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace dd {
namespace {

fp densePauliExpectation(const CVec& state, const llvm::StringRef paulis) {
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
  auto packageOwner = mqt::test::value(Package::create(3));
  auto& package = *packageOwner;
  CVec amplitudes{
      {.1, .2},  {.3, -.4}, {-.2, .3}, {.4, .1},
      {.2, -.1}, {-.3, .2}, {.1, -.3}, {.4, -.2},
  };
  fp norm = 0.;
  for (const auto amplitude : amplitudes) {
    norm += std::norm(amplitude);
  }
  for (auto& amplitude : amplitudes) {
    amplitude /= std::sqrt(norm);
  }
  const auto state = mqt::test::value(makeStateFromVector(amplitudes, package));
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
  const auto values =
      mqt::test::value(package.expectationValues(state, observables));
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
  package.decRef(state);
  EXPECT_TRUE(package.getRootSet<vNode>().empty());
}

TEST(PauliExpectations, InvalidInputsLeaveBorrowedStateLive) {
  auto packageOwner = mqt::test::value(Package::create(2));
  auto& package = *packageOwner;
  const auto state = mqt::test::value(makeZeroState(2, package));
  const auto roots = package.getRootSet<vNode>();
  const std::array<PauliObservable, 1> valid{
      PauliObservable{{.paulis = "II", .coefficient = 1.}},
  };
  for (const auto& term : std::vector<PauliTerm>{
           {.paulis = "X", .coefficient = 1.},
           {.paulis = "IA", .coefficient = 1.},
           {
               .paulis = "II",
               .coefficient = std::numeric_limits<fp>::quiet_NaN(),
           },
       }) {
    const std::array<PauliObservable, 1> invalid{PauliObservable{term}};
    EXPECT_EQ(mqt::test::errorCode(
                  [&] { return package.expectationValues(state, invalid); }),
              std::make_error_code(std::errc::invalid_argument));
  }
  auto otherOwner = mqt::test::value(Package::create(2));
  auto& other = *otherOwner;
  EXPECT_EQ(mqt::test::errorCode(
                [&] { return other.expectationValues(state, valid); }),
            std::make_error_code(std::errc::invalid_argument));
  EXPECT_EQ(package.getRootSet<vNode>(), roots);
  EXPECT_TRUE(package.getRootSet<mNode>().empty());
  package.decRef(state);
  EXPECT_EQ(mqt::test::errorCode(
                [&] { return package.expectationValues(state, valid); }),
            std::make_error_code(std::errc::invalid_argument));
}

} // namespace
} // namespace dd
