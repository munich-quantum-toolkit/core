from mqt.core.xdsl.dialects import get_all_dialects
from mqt.core.xdsl.transforms import get_all_passes
from xdsl.xdsl_opt_main import xDSLOptMain


class MQTCoreOptMain(xDSLOptMain):
    def register_all_dialects(self):
        # First register all standard xDSL dialects from the multiverse
        super().register_all_dialects()
        # Then register our custom dialects
        for name, dialect_func in get_all_dialects().items():
            self.ctx.register_dialect(name, dialect_func)

    def register_all_passes(self):
        for name, pass_ in get_all_passes().items():
            self.register_pass(name, pass_)

    def register_all_targets(self):
        super().register_all_targets()


def main():
    mqtcore_main = MQTCoreOptMain()
    mqtcore_main.run()


if "__main__" == __name__:
    main()
