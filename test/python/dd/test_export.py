# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared vector and matrix DD exports."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.core.dd import DDPackage


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("colored", [False, True])
@pytest.mark.parametrize("classic", [False, True])
@pytest.mark.parametrize("edge_labels", [False, True])
def test_dot_options(*, matrix: bool, colored: bool, classic: bool, edge_labels: bool) -> None:
    """Keep export options available for both DD types."""
    package = DDPackage(1)
    dd = (
        package.from_matrix(np.array([[1, 1j], [1, -1j]]) / np.sqrt(2))
        if matrix
        else package.from_vector(np.array([1, 1j]) / np.sqrt(2))
    )
    dot = dd.to_dot(colored=colored, classic=classic, edge_labels=edge_labels)
    edges = "\n".join(line for line in dot.splitlines() if "->" in line)
    assert (' color="' in edges) == colored
    assert ("shape=circle" in dot) == classic
    assert ('label=<<font point-size="8">' in edges) == edge_labels
