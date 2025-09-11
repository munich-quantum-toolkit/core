#!/usr/bin/env python3
# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import sys
from pathlib import Path

# Add the parent directory to Python path so we can import from xdsl/dialects
sys.path.insert(0, str(Path(__file__).parent))

from dialects import get_all_dialects
from xdsl.xdsl_opt_main import xDSLOptMain


class QuoptMain(xDSLOptMain):
    def register_all_dialects(self) -> None:
        dialects = get_all_dialects()
        for name, dialect_func in dialects.items():
            dialect_func()
            self.ctx.register_dialect(name, dialect_func)


def main() -> None:
    quopt_main = QuoptMain()
    quopt_main.register_all_dialects()

    # Test loading the mqtopt dialect
    mqtopt_dialect = quopt_main.ctx.get_dialect("mqtopt")
    if mqtopt_dialect:
        pass


if __name__ == "__main__":
    main()
