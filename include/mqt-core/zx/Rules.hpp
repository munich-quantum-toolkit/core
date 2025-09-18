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

#include "ZXDefinitions.hpp"

namespace zx {
class ZXDiagram;

/**
 * @brief Check whether the spider has exactly two incident edges and a phase of
 * 0.
 * @param diag The diagram.
 * @param v The vertex to check.
 * @return True if the spider has exactly two incident edges and a phase of 0.
 */

bool checkIdSimp(const ZXDiagram& diag, Vertex v);

/**
 * @brief Remove spider from the diagram. checkIdSimp() should yield true.
 * @details The spider is removed by connecting its incident edges.
 */
void removeId(ZXDiagram& diag, Vertex v);

/**
 * @brief Check whether two spiders can be fused.
 * @details See https://arxiv.org/abs/2012.13966, page 27, rule (f).
 * @param diag The diagram.
 * @param v0 The first spider.
 * @param v1 The second spider.
 * @return True if the spiders can be fused.
 */
bool checkSpiderFusion(const ZXDiagram& diag, Vertex v0, Vertex v1);

/**
 * @brief Fuse two spiders. checkSpiderFusion() should yield true.
 * @details The spiders are fused by connecting their incident edges.
 */
void fuseSpiders(ZXDiagram& diag, Vertex v0, Vertex v1);

/**
 * @brief Check if the local complementation rule can be applied to the given
 * spider.
 * @details See https://arxiv.org/abs/2012.13966, equation (102).
 * @param diag The diagram.
 * @param v The spider.
 * @return True if the local complementation rule can be applied.
 */
bool checkLocalComp(const ZXDiagram& diag, Vertex v);

/**
 * @brief Apply the local complementation rule to the given spider.
 * checkLocalComp() should yield true.
 * @details See https://arxiv.org/abs/2012.13966, equation (102).
 */
void localComp(ZXDiagram& diag, Vertex v);

/**
 * @brief Check if the pivot rule can be applied to the given interior spiders.
 * @details See https://arxiv.org/abs/2012.13966, equation (103).
 * @param diag The diagram.
 * @param v0 The first spider.
 * @param v1 The second spider.
 * @return True if the pivot rule can be applied.
 */
bool checkPivotPauli(const ZXDiagram& diag, Vertex v0, Vertex v1);

/**
 * @brief Apply the pivot rule to the given interior spiders. checkPivotPauli()
 * should yield true.
 * @details See https://arxiv.org/abs/2012.13966, equation (103).
 */
void pivotPauli(ZXDiagram& diag, Vertex v0, Vertex v1);

/**
 * @brief Check if the pivot rule can be applied. Spiders can be connected to
 * boundaries.
 * @param diag The diagram.
 * @param v0 The first spider.
 * @param v1 The second spider.
 * @return True if the pivot rule can be applied.
 */
bool checkPivot(const ZXDiagram& diag, Vertex v0, Vertex v1);

/**
 * @brief Apply the pivot rule. Spiders can be connected to boundaries.
 * checkPivot() should yield true.
 */
void pivot(ZXDiagram& diag, Vertex v0, Vertex v1);

/**
 * @brief Check if the gadget pivot rule can be applied. Spiders can be
 * connected to boundaries.
 * @details See https://arxiv.org/abs/1903.10477, page 13
 * rule (P2) and (P3) for details.
 * @param diag The diagram.
 * @param v0 The first spider.
 * @param v1 The second spider.
 * @return True if the gadget pivot rule can be applied.
 */
bool checkPivotGadget(const ZXDiagram& diag, Vertex v0, Vertex v1);

/**
 * @brief Apply the gadget pivot rule. Spiders can be connected to boundaries.
 * checkPivotGadget() should yield true.
 * @details See https://arxiv.org/abs/1903.10477, page 13
 * rule (P2) and (P3) for details.
 */
void pivotGadget(ZXDiagram& diag, Vertex v0, Vertex v1);

/**
 * @brief Check if a gadget can be fused with its connected spider and fuse if
 * true.
 * @details Unlike other rules, this function performs the check and
 * modification in one step. This is for performance reasons, since the overhead
 * for the check is significant and otherwise intermediate results would have to
 * be computed twice. See https://arxiv.org/abs/1903.10477, page 13, rule (ID).
 * @param diag The diagram.
 * @param v The spider.
 * @return True if the gadget can be fused with its connected spider.
 */
bool checkAndFuseGadget(ZXDiagram& diag, Vertex v);
} // namespace zx
