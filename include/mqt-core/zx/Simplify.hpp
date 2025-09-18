/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "Rules.hpp"
#include "ZXDefinitions.hpp"
#include "ZXDiagram.hpp"

#include <cstddef>

namespace zx {

/**
 * @brief Simplify the diagram by applying the given rule to all vertices that
 * match the given check.
 * @tparam VertexCheckFun Type of the check function
 * @tparam VertexRuleFun Type of the rule function
 * @param diag The diagram to simplify
 * @param check The check function that determines if a vertex should be
 * simplified
 * @param rule The rule function that is applied to the vertex
 * @return The number of simplifications that were applied
 */
template <class VertexCheckFun, class VertexRuleFun>
std::size_t simplifyVertices(ZXDiagram& diag, VertexCheckFun check,
                             VertexRuleFun rule) {
  std::size_t nSimplifications = 0;
  bool newMatches = true;

  while (newMatches) {
    newMatches = false;
    for (const auto& [v, _] : diag.getVertices()) {
      if (check(diag, v)) {
        rule(diag, v);
        newMatches = true;
        nSimplifications++;
      }
    }
  }

  return nSimplifications;
}

/**
 * @brief Simplify the diagram by applying the given rule to all edges that
 * match the given check.
 * @tparam EdgeCheckFun Type of the check function
 * @tparam EdgeRuleFun Type of the rule function
 * @param diag The diagram to simplify
 * @param check The check function that determines if an edge should be
 * simplified
 * @param rule The rule function that is applied to the edge
 * @return The number of simplifications that were applied
 */
template <class EdgeCheckFun, class EdgeRuleFun>
std::size_t simplifyEdges(ZXDiagram& diag, EdgeCheckFun check,
                          EdgeRuleFun rule) {
  std::size_t nSimplifications = 0;
  bool newMatches = true;

  while (newMatches) {
    newMatches = false;
    for (const auto& [v0, v1] : diag.getEdges()) {
      if (diag.isDeleted(v0) || diag.isDeleted(v1) || !check(diag, v0, v1)) {
        continue;
      }
      rule(diag, v0, v1);
      newMatches = true;
      nSimplifications++;
    }
  }

  return nSimplifications;
}

std::size_t gadgetSimp(ZXDiagram& diag);

/**
 * @brief Apply the identity rule to the Diagram until exhaustion.
 * @details A spider with exactly two incident edges and a phase of 0 is
 * removed. See https://arxiv.org/abs/2012.13966 page 27, rule (id).
 * @param diag The diagram to simplify
 * @return The number of simplifications that were applied
 */

std::size_t idSimp(ZXDiagram& diag);

/**
 * @brief Apply the spider rule to the Diagram until exhaustion.
 * @details Spiders of the same type connected by a regular edge can be turned
 * into one
 * by adding both phases and fusing all incident edges.
 * See https://arxiv.org/abs/2012.13966 page 27, rule (f).
 * @param diag The diagram to simplify
 * @return The number of simplifications that were applied
 */
std::size_t spiderSimp(ZXDiagram& diag);

/**
 * @brief Apply the local complementation rule to the Diagram until exhaustion.
 * @details See https://arxiv.org/abs/2012.13966 equation (102).
 * @param diag The diagram to simplify
 * @return The number of simplifications that were applied
 */
std::size_t localCompSimp(ZXDiagram& diag);

/**
 * @brief Simplify the diagram by applying the pivot rule.
 * @details See https://arxiv.org/abs/2012.13966, equation 103 for details.
 * Similar to the Pauli pivot rule but can be applied to vertices connectedto
 * the boundary.
 * @param diag The diagram to simplify.
 * @return The number of simplifications applied.
 */
std::size_t pivotSimp(ZXDiagram& diag);

/**
 * Simplify the diagram by applying the pivot rule in the case that the pivot
 * spiders have phases of Pi. See https://arxiv.org/abs/2012.13966, equation 103
 * for details.
 * @param diag The diagram to simplify.
 * @return The number of simplifications applied.
 */
std::size_t pivotPauliSimp(ZXDiagram& diag);

/**
 * @brief Simplify all internal vertices and edges of the diagram using Clifford
 * simplifications.
 * @details This function applies the id, spider, local complementation,
 * and Pauli pivot rules.
 * @param diag The diagram to simplify.
 * @return The number of simplifications applied.
 */
std::size_t interiorCliffordSimp(ZXDiagram& diag);

/*
 * @brief Simplify the diagram using Clifford simplifications.
 * @details In addition to interior Clifford simplifications, this function also
 * applies the regular pivot rule.
 * @param diag The diagram to simplify.
 * @return The number of simplifications applied.
 */
std::size_t cliffordSimp(ZXDiagram& diag);

/**
 * @brief Simplify the diagram by applying the pivot rule to non-Pauli spider.
 * @details By extracting the phases into extra "gadgets", the pivot rule can be
 * applies to non-Pauli spiders. See https://arxiv.org/abs/1903.10477, page 13
 * rule (P2) and (P3) for details.
 * @param diag The diagram to simplify.
 * @return The number of simplifications applied.
 */
std::size_t pivotgadgetSimp(ZXDiagram& diag);

/**
 * @brief Simplify the diagram by applying Clifford simplifications and the
 * gadget pivot rule.
 * @details In addition to the Clifford simplification, this function also
 * applies the pivot gadget rule.
 * @param diag The diagram to simplify.
 * @return The number of simplifications applied.
 */
std::size_t fullReduce(ZXDiagram& diag);

/**
 * @brief Apply full reduction to the diagram. Rounds phases to nearest multiple
 * of Pi/2 during simplification.
 * @param diag The diagram to simplify.
 * @param tolerance The tolerance for rounding phases to multiples of Pi/2.
 * @return The number of simplifications applied
 */
std::size_t fullReduceApproximate(ZXDiagram& diag, fp tolerance);

} // namespace zx
