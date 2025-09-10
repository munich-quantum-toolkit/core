#!/usr/bin/env python3
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import from xdsl/dialects
sys.path.insert(0, str(Path(__file__).parent))

from dialects import get_all_dialects
from xdsl.xdsl_opt_main import xDSLOptMain


class QuoptMain(xDSLOptMain):
    def register_all_dialects(self):
        dialects = get_all_dialects()
        print(f"Registering {len(dialects)} dialects:")
        for name, dialect_func in dialects.items():
            dialect = dialect_func()
            print(f"  {name}: {dialect}")
            print(f"    Operations: {[op.name for op in dialect.operations]}")
            print(f"    Types: {[t.name for t in dialect.attributes]}")
            self.ctx.register_dialect(name, dialect_func)
            print(f"    Registered successfully!")


def main():
    print("Testing dialect registration...")
    quopt_main = QuoptMain()
    quopt_main.register_all_dialects()
    
    print(f"\nContext dialects: {list(quopt_main.ctx.dialects.keys())}")
    
    # Test loading the mqtopt dialect
    mqtopt_dialect = quopt_main.ctx.get_dialect("mqtopt")
    print(f"MQTOpt dialect: {mqtopt_dialect}")
    if mqtopt_dialect:
        print(f"  Operations: {[op.name for op in mqtopt_dialect.operations]}")
        print(f"  Types: {[t.name for t in mqtopt_dialect.attributes]}")


if __name__ == "__main__":
    main()
