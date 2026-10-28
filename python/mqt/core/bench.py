# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Public bench module backed by the shared native extension."""

import sys

from mqt.core import _core

bench = _core.bench
bench.__file__ = _core.__file__
sys.modules[__name__] = bench
