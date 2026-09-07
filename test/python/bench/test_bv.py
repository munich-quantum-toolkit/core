# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the bv benchmark."""

from __future__ import annotations

from mqt.core.bench import bv

from .utils import assert_generates


def test_bv_methods_share_the_hidden_string_reference() -> None:
    """Expose static and dynamic Bernstein--Vazirani as one family."""
    for method in (bv.Method.STATIC, bv.Method.DYNAMIC):
        benchmark = bv.BV(bv.Options(hidden_bitstring="101", method=method))
        assert benchmark.probability("101") == 1
        assert benchmark.evaluate({"101": 10}).success_probability == 1
        assert bv.BV.from_manifest_json(benchmark.manifest_json).case_id == benchmark.case_id
        assert_generates(benchmark.generate())
