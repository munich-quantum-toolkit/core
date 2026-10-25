# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Public module names share the native type registry."""

from importlib import import_module

import pytest


@pytest.mark.parametrize("name", ["bench", "dd", "qdmi", "mlir"])
def test_public_native_module_identity(name: str) -> None:
    """The public import exposes the same module and native types."""
    core = import_module("mqt.core._core")
    public = import_module(f"mqt.core.{name}")
    assert public is getattr(core, name)
    assert public.__name__ == f"mqt.core.{name}"
    assert public.__file__ == core.__file__
