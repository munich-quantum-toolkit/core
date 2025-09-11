# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from xdsl.xdsl_opt_main import xDSLOptMain

from mqt.core.xdsl.dialects import get_all_dialects
from mqt.core.xdsl.transforms import get_all_passes


class MQTCoreOptMain(xDSLOptMain):
    def register_all_dialects(self) -> None:
        # First register all standard xDSL dialects from the multiverse
        super().register_all_dialects()
        # Then register our custom dialects
        for name, dialect_func in get_all_dialects().items():
            self.ctx.register_dialect(name, dialect_func)

    def register_all_passes(self) -> None:
        for name, pass_ in get_all_passes().items():
            self.register_pass(name, pass_)

    def register_all_targets(self) -> None:
        super().register_all_targets()


def main() -> None:
    mqtcore_main = MQTCoreOptMain()
    mqtcore_main.run()


if __name__ == "__main__":
    main()
