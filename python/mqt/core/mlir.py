# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Public mlir module backed by the shared native extension."""

import sys
from importlib import import_module

from mqt.core import _core

mlir = _core.mlir
mlir.__file__ = _core.__file__
sys.modules[__name__] = mlir

# Stage installed providers before compiler device resolution.
import_module("mqt.core.qdmi")
