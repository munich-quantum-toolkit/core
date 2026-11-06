/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/MatrixComparison.hpp"

#include "dd/ComplexValue.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <unordered_map>
#include <vector>

namespace dd {
namespace {
/// Normalized trace and RMS distance from that scalar identity.
struct FrobeniusSummary {
  std::complex<double> mean{};
  double residual = 0.;

  [[nodiscard]] double norm() const {
    return std::hypot(residual, std::abs(mean));
  }
  [[nodiscard]] double distanceToIdentity() const {
    return std::hypot(residual, std::abs(mean - 1.));
  }
  [[nodiscard]] double distanceUpToPhase() const {
    return std::hypot(residual, std::abs(mean) - 1.);
  }
};

/// Cache summaries without incoming edge weights for one comparison.
class FrobeniusTraversal {
public:
  [[nodiscard]] FrobeniusSummary summarize(const mEdge& edge) {
    const auto value = static_cast<ComplexValue>(edge.w);
    const std::complex<double> weight{value.r, value.i};
    if (weight == 0.) {
      return {};
    }
    if (edge.isTerminal()) {
      return {.mean = weight, .residual = 0.};
    }
    const auto summary = summarizeNode(edge.p);
    return {
        .mean = weight * summary.mean,
        .residual = std::abs(weight) * summary.residual,
    };
  }

private:
  std::unordered_map<const mNode*, FrobeniusSummary> cache;

  FrobeniusSummary summarizeNode(const mNode* node) {
    if (const auto it = cache.find(node); it != cache.end()) {
      return it->second;
    }
    const auto a = summarize(node->e[0]);
    const auto b = summarize(node->e[1]);
    const auto c = summarize(node->e[2]);
    const auto d = summarize(node->e[3]);
    const auto within = std::hypot(std::hypot(a.residual, d.residual),
                                   std::hypot(b.norm(), c.norm())) /
                        std::numbers::sqrt2;
    const FrobeniusSummary result{
        .mean = (a.mean + d.mean) * 0.5,
        .residual = std::hypot(within, std::abs(a.mean - d.mean) * 0.5),
    };
    cache.emplace(node, result);
    return result;
  }
};

/// Reject using a bounded sum of disjoint off-diagonal terminal blocks.
/// Requires finite weights, IEEE arithmetic, and gradual underflow.
class BoundedFrobeniusTraversal {
public:
  [[nodiscard]] bool rejects(const mEdge& edge, const double tolerance,
                             const std::vector<bool>& garbage) {
    const auto limit = std::nextafter(tolerance * tolerance,
                                      std::numeric_limits<double>::infinity());
    return !edge.isTerminal() && probe(edge, 1., false, limit, garbage);
  }

private:
  std::size_t probes = 0;
  double mass = 0.;

  // Round nonnegative arithmetic down, including overflow and underflow.
  static double lower(const double value) {
    return value == 0. ? 0. : std::nextafter(value, 0.);
  }

  bool probe(const mEdge& edge, const double scale, const bool offDiagonal,
             const double limit, const std::vector<bool>& garbage) {
    if (probes == 32) {
      return false;
    }
    ++probes;
    const auto weight = static_cast<ComplexValue>(edge.w);
    const auto squared =
        lower(lower(weight.r * weight.r) + lower(weight.i * weight.i));
    const auto contribution = lower(scale * squared);
    if (contribution == 0.) {
      return false;
    }
    if (edge.isTerminal()) {
      if (offDiagonal) {
        mass = lower(mass + contribution);
      }
      return mass > limit;
    }
    const auto childScale = lower(contribution * 0.5);
    const auto measured = edge.p->v >= garbage.size() || !garbage[edge.p->v];
    return std::ranges::any_of(
        std::array{1U, 2U, 0U, 3U}, [&](const auto index) {
          return probe(edge.p->e[index], childScale,
                       offDiagonal || (measured && (index == 1 || index == 2)),
                       limit, garbage);
        });
  }
};

/// RMS matrix norm and RMS weight outside measured-output diagonal blocks.
struct PartialSummary {
  double norm = 0.;
  double leakage = 0.;
};

/// Compute the weight outside measured-output diagonal blocks.
class PartialTraversal {
public:
  explicit PartialTraversal(const std::vector<bool>& garbageQubits)
      : garbage(garbageQubits) {}

  [[nodiscard]] PartialSummary summarize(const mEdge& edge) {
    const auto weight = static_cast<ComplexValue>(edge.w).mag();
    if (weight == 0.) {
      return {};
    }
    if (edge.isTerminal()) {
      return {.norm = weight, .leakage = 0.};
    }
    const auto summary = summarizeNode(edge.p);
    return {.norm = weight * summary.norm, .leakage = weight * summary.leakage};
  }

private:
  std::vector<bool> garbage;
  std::unordered_map<const mNode*, PartialSummary> cache;

  PartialSummary summarizeNode(const mNode* node) {
    if (const auto it = cache.find(node); it != cache.end()) {
      return it->second;
    }
    const auto a = summarize(node->e[0]);
    const auto b = summarize(node->e[1]);
    const auto c = summarize(node->e[2]);
    const auto d = summarize(node->e[3]);
    const auto ignored = node->v < garbage.size() && garbage[node->v];
    const PartialSummary result{
        .norm =
            std::hypot(std::hypot(a.norm, d.norm), std::hypot(b.norm, c.norm)) /
            std::numbers::sqrt2,
        .leakage = std::hypot(std::hypot(a.leakage, d.leakage),
                              std::hypot(ignored ? b.leakage : b.norm,
                                         ignored ? c.leakage : c.norm)) /
                   std::numbers::sqrt2,
    };
    cache.emplace(node, result);
    return result;
  }
};

} // namespace

IdentityComparison compareToIdentity(const mEdge& matrix, const fp tolerance) {
  if (BoundedFrobeniusTraversal{}.rejects(matrix, tolerance, {})) {
    return {};
  }
  const auto summary = FrobeniusTraversal{}.summarize(matrix);
  return {
      .closeToIdentity = summary.distanceToIdentity() <= tolerance,
      .closeUpToGlobalPhase = summary.distanceUpToPhase() <= tolerance,
  };
}

bool isCloseToDiagonal(const mEdge& matrix, const fp tolerance,
                       const std::vector<bool>& garbage) {
  if (BoundedFrobeniusTraversal{}.rejects(matrix, tolerance, garbage)) {
    return false;
  }
  return PartialTraversal(garbage).summarize(matrix).leakage <= tolerance;
}
} // namespace dd
